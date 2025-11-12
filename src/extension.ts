import * as vscode from 'vscode';
import { TritonCompiler } from './tritonCompiler';
import { AssemblyViewerPanel } from './assemblyViewer';
import { HighlightManager } from './highlighting';
import { logger } from './logger';

export function activate(context: vscode.ExtensionContext) {
    logger.log('Extension activated');
    // logger.show(); // Uncomment to show output panel for debugging

    const compiler = new TritonCompiler();
    const highlightManager = new HighlightManager();

    // Restore assembly viewer if it was open before reload
    const savedState = context.workspaceState.get<{ documentUri: string, line: number }>('assemblyViewerState');
    if (savedState) {
        logger.log('Restoring assembly viewer from previous session');
        try {
            const uri = vscode.Uri.parse(savedState.documentUri);
            vscode.workspace.openTextDocument(uri).then(
                document => {
                    AssemblyViewerPanel.createOrShow(context.extensionUri, compiler, document, highlightManager, savedState.line, context);
                },
                (err: Error) => {
                    logger.log('Failed to restore assembly viewer: ' + err.message);
                    context.workspaceState.update('assemblyViewerState', undefined);
                }
            );
        } catch (err) {
            logger.log('Failed to parse saved state: ' + (err instanceof Error ? err.message : String(err)));
            context.workspaceState.update('assemblyViewerState', undefined);
        }
    }

    // Command: Show Assembly Side by Side
    const showAssemblySideBySide = vscode.commands.registerCommand(
        'trident.showAssemblySideBySide',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            if (document.languageId !== 'python') {
                vscode.window.showWarningMessage('This command only works with Python files');
                return;
            }

            // Get the current line number where cursor is
            const currentLine = editor.selection.active.line;

            AssemblyViewerPanel.createOrShow(context.extensionUri, compiler, document, highlightManager, currentLine, context);
        }
    );


    // Auto-highlight assembly when clicking on source code
    let selectionTimeout: NodeJS.Timeout | undefined;
    const onSelectionChangeHandler = vscode.window.onDidChangeTextEditorSelection(async (event) => {
        const editor = event.textEditor;
        const document = editor.document;

        // Only process Python files
        if (document.languageId !== 'python') {
            return;
        }

        // Only highlight if assembly viewer is already open
        if (!AssemblyViewerPanel.currentPanel) {
            return;
        }

        // Skip if this is a programmatic selection (from clicking assembly)
        // This prevents circular highlighting: assembly->source->assembly
        if (AssemblyViewerPanel.isProgrammaticSelection()) {
            logger.log('Skipping highlight - programmatic selection');
            return;
        }

        // Debounce to avoid excessive updates while moving cursor
        if (selectionTimeout) {
            clearTimeout(selectionTimeout);
        }

        selectionTimeout = setTimeout(() => {
            const currentLine = editor.selection.active.line;
            logger.log(`Selection changed to line ${currentLine} in ${document.fileName}`);

            // Clear previous custom highlights (from clicking on assembly)
            highlightManager.clearHighlights();

            // Highlight the assembly for the current line
            AssemblyViewerPanel.currentPanel?.highlightAssemblyForSourceLine(document, currentLine);
        }, 150); // 150ms debounce
    });

    context.subscriptions.push(showAssemblySideBySide, onSelectionChangeHandler, highlightManager);
}

export function deactivate() { }

