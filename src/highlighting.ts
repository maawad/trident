import * as vscode from 'vscode';

/**
 * Manages bidirectional highlighting between source code and assembly
 */
export class HighlightManager {
    private sourceDecorations: vscode.TextEditorDecorationType;
    private assemblyHighlightCallback?: (lines: number[]) => void;
    private lastDecoratedEditor?: vscode.TextEditor;

    constructor() {
        this.sourceDecorations = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('editor.selectionBackground'),
            border: '1px solid',
            borderColor: new vscode.ThemeColor('editor.selectionHighlightBorder'),
            isWholeLine: true
        });
    }

    /**
     * Register callback for assembly highlighting
     */
    registerAssemblyHighlightCallback(callback: (lines: number[]) => void) {
        this.assemblyHighlightCallback = callback;
    }

    /**
     * Highlight source lines and notify assembly viewer
     */
    highlightSource(editor: vscode.TextEditor, lines: number[]) {
        // Clear highlights from the previous editor if it's different
        if (this.lastDecoratedEditor && this.lastDecoratedEditor !== editor) {
            this.lastDecoratedEditor.setDecorations(this.sourceDecorations, []);
        }

        const ranges = lines.map(line =>
            editor.document.lineAt(Math.min(line, editor.document.lineCount - 1)).range
        );

        editor.setDecorations(this.sourceDecorations, ranges);
        this.lastDecoratedEditor = editor;

        // Notify assembly viewer to highlight corresponding lines
        if (this.assemblyHighlightCallback) {
            const assemblyLines = this.mapSourceToAssembly(lines);
            this.assemblyHighlightCallback(assemblyLines);
        }
    }

    /**
     * Highlight source from assembly line click
     */
    highlightFromAssembly(assemblyLine: number) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        // Map assembly line to source lines
        const sourceLines = this.mapAssemblyToSource(assemblyLine);

        if (sourceLines.length > 0) {
            this.highlightSource(editor, sourceLines);

            // Scroll to first line
            const range = editor.document.lineAt(sourceLines[0]).range;
            editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
        }
    }

    /**
     * Clear all highlights
     */
    clearHighlights() {
        // Clear decorations from all visible editors
        vscode.window.visibleTextEditors.forEach(editor => {
            editor.setDecorations(this.sourceDecorations, []);
        });
        this.lastDecoratedEditor = undefined;
    }

    /**
     * Map source lines to assembly lines
     * TODO: Parse debug info from assembly to create accurate mapping
     */
    private mapSourceToAssembly(sourceLines: number[]): number[] {
        // Placeholder: In real implementation, parse LOC directives from GCN assembly
        // .loc 1 42 0  ; Line 42, column 0, file 1
        return sourceLines;
    }

    /**
     * Map assembly line to source lines
     * TODO: Parse debug info from assembly to create accurate mapping
     */
    private mapAssemblyToSource(assemblyLine: number): number[] {
        // Placeholder: Parse .loc directives to map back to source
        return [assemblyLine];
    }

    dispose() {
        this.sourceDecorations.dispose();
    }
}

/**
 * Parse GCN assembly to extract debug location information
 */
export function parseDebugInfo(assembly: string): Map<number, number[]> {
    const mapping = new Map<number, number[]>();
    const lines = assembly.split('\n');

    let currentSourceLine = -1;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Parse .loc directive: .loc <file> <line> <column>
        const locMatch = line.match(/\.loc\s+\d+\s+(\d+)\s+\d+/);
        if (locMatch) {
            currentSourceLine = parseInt(locMatch[1]) - 1; // Convert to 0-indexed
        }

        // Map assembly line to source line
        if (currentSourceLine >= 0) {
            if (!mapping.has(currentSourceLine)) {
                mapping.set(currentSourceLine, []);
            }
            mapping.get(currentSourceLine)!.push(i);
        }
    }

    return mapping;
}

