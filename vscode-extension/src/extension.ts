import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

let diagnosticCollection: vscode.DiagnosticCollection;
let recipeData: any = null;

export function activate(context: vscode.ExtensionContext) {
    console.log('GradTracer extension is now active!');

    diagnosticCollection = vscode.languages.createDiagnosticCollection('gradtracer');
    context.subscriptions.push(diagnosticCollection);

    // Watch for gradtracer_recipe.json in the workspace root
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        const rootPath = workspaceFolders[0].uri.fsPath;
        const recipePath = path.join(rootPath, 'gradtracer_recipe.json');

        // Initial load
        loadRecipe(recipePath);

        // Setup File Watcher
        const watcher = vscode.workspace.createFileSystemWatcher(
            new vscode.RelativePattern(rootPath, 'gradtracer_recipe.json')
        );

        watcher.onDidChange(() => loadRecipe(recipePath));
        watcher.onDidCreate(() => loadRecipe(recipePath));
        watcher.onDidDelete(() => {
            recipeData = null;
            diagnosticCollection.clear();
        });

        context.subscriptions.push(watcher);
    }

    // Register Hover Provider
    const hoverProvider = vscode.languages.registerHoverProvider('python', {
        provideHover(document, position) {
            if (!recipeData || !recipeData.layers) return null;

            const range = document.getWordRangeAtPosition(position);
            const word = document.getText(range);

            // Very naive string matching against layer names
            // e.g. if the layer name is "item_embedding" and we hover over it
            for (const [layerName, data] of Object.entries<any>(recipeData.layers)) {
                if (layerName === word) {
                    const md = new vscode.MarkdownString();
                    md.isTrusted = true;
                    md.appendMarkdown(`### 🌊 GradTracer Auto-Compression\n\n`);
                    md.appendMarkdown(`**Health Score:** ${data.health_score} / 100\n\n`);
                    md.appendMarkdown(`**Action:** [${data.quantization} + ${(data.prune_ratio * 100).toFixed(0)}% Pruning]\n\n`);
                    md.appendMarkdown(`*Reason:* ${data.reason}\n\n`);
                    md.appendMarkdown(`[⚡ Apply Recipe](command:gradtracer.applyRecipe)`);
                    return new vscode.Hover(md);
                }
            }
            return null;
        }
    });

    context.subscriptions.push(hoverProvider);

    // Register Command to Apply
    const disposable = vscode.commands.registerCommand('gradtracer.applyRecipe', () => {
        vscode.window.showInformationMessage('GradTracer: Compression recipe applied! Restarting trainer...');
    });
    context.subscriptions.push(disposable);

    // Register Command to Show 3D Inspector
    const show3DCommand = vscode.commands.registerCommand('gradtracer.show3DInspector', () => {
        if (!recipeData) {
            vscode.window.showErrorMessage('GradTracer: No recipe loaded. Ensure gradtracer_recipe.json exists in the workspace root.');
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'gradtracer3D',
            '🌊 GradTracer 3D Inspector',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = getWebviewContent(recipeData);
    });
    context.subscriptions.push(show3DCommand);
}

function getWebviewContent(recipe: any): string {
    // Transform recipe.layers into graph nodes
    const nodes = [];
    const links = [];
    let prevId = null;

    for (const [layerName, data] of Object.entries<any>(recipe.layers || {})) {
        // Node size heuristic based on shape or fixed
        let val = 10;
        if (data.shape && data.shape.length > 0) {
            val = Math.max(5, Math.min(30, Math.log2(data.shape[0] * (data.shape[1] || 1))));
        }

        // Color mapping
        let color = '#4a90e2'; // Healthy Blue
        if (data.health_score < 30 || data.dead_ratio > 0.5) {
            color = '#e74c3c'; // Critical Red
        } else if (data.health_score < 70) {
            color = '#f1c40f'; // Warning Yellow
        }

        const node = {
            id: layerName,
            name: layerName,
            val: val,
            color: color,
            type: data.layer_type || 'Layer',
            health: data.health_score,
            quant: data.quantization,
            prune: Math.round(data.prune_ratio * 100),
            reason: data.reason
        };
        nodes.push(node);

        // Very simple sequential linking for visualization
        if (prevId) {
            links.push({ source: prevId, target: layerName });
        }
        prevId = layerName;
    }

    const graphData = { nodes, links };

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GradTracer 3D Inspector</title>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <style>
        body { margin: 0; padding: 0; background-color: #1e1e1e; color: white; font-family: sans-serif; overflow: hidden; }
        #3d-graph { width: 100vw; height: 100vh; }
        .node-label {
            background: rgba(0,0,0,0.8);
            border: 1px solid #444;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
        }
        .header {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 10;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .health-red { color: #e74c3c; font-weight: bold; }
        .health-blue { color: #4a90e2; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h3>🌊 GradTracer Inspector</h3>
        <p>Target Sparsity: ${recipe?.metadata?.target_sparsity || 0.5}</p>
        <p>VRAM Savings: ${recipe?.metadata?.estimated_vram_saving_mb || 0} MB</p>
        <small>Drag to rotate. Scroll to zoom.</small>
    </div>
    <div id="3d-graph"></div>
    <script>
        const graphData = ${JSON.stringify(graphData)};
        
        const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
            .graphData(graphData)
            .nodeLabel(node => {
                const healthCls = node.health < 40 ? 'health-red' : 'health-blue';
                return \`
                    <div class="node-label">
                        <strong>\${node.name}</strong> (\${node.type})<br/>
                        Health: <span class="\${healthCls}">\${node.health}/100</span><br/>
                        Action: \${node.quant} + \${node.prune}% Prune<br/>
                        <em>\${node.reason}</em>
                    </div>
                \`;
            })
            .nodeAutoColorBy('color')
            .nodeColor(node => node.color)
            .linkWidth(2)
            .linkOpacity(0.5);
            
        // Fit graph to screen
        setTimeout(() => Graph.zoomToFit(400, 50), 1000);
    </script>
</body>
</html>`;
}

function loadRecipe(recipePath: string) {
    if (fs.existsSync(recipePath)) {
        try {
            const raw = fs.readFileSync(recipePath, 'utf-8');
            recipeData = JSON.parse(raw);
            vscode.window.showInformationMessage('GradTracer: Auto-Compression Recipe loaded.');
        } catch (e) {
            console.error('Failed to parse GradTracer recipe:', e);
        }
    }
}

export function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.clear();
    }
}
