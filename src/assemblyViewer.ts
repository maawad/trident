import * as vscode from 'vscode';
import * as path from 'path';
import { TritonCompiler, KernelAssembly, IRType } from './tritonCompiler';
import { HighlightManager } from './highlighting';
import { logger } from './logger';
import { LineMappingBuilder, LineMapping } from './lineMapping';

interface FilterState {
    hideDirectives: boolean;
    hideComments: boolean;
    hideEmpty: boolean;
    hiddenInstructions: string[];
}

export class AssemblyViewerPanel {
    public static currentPanel: AssemblyViewerPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _compiler: TritonCompiler;
    private _highlightManager: HighlightManager;
    private _currentDocument: vscode.TextDocument | undefined;
    private _assemblies: KernelAssembly[] = [];
    private _allAssemblies: KernelAssembly[] = []; // Keep all assemblies for filtering
    private _lineMapping: LineMapping = { sourceToAsm: new Map(), asmLineToFile: new Map() }; // Unified line mapping
    private _mappingBuilder: LineMappingBuilder = new LineMappingBuilder();
    private _selectedSourceFile: string | undefined;
    private _selectedKernelIndex: number | undefined; // Track selected kernel for single-view
    private _selectedIRType: IRType = 'gcn'; // Default to GCN
    private _allOpenTritonFiles: Set<string> = new Set();
    private _diffState: { kernel1: KernelAssembly, kernel2: KernelAssembly, index1: number, index2: number } | undefined;
    private _themeChangeTimeout: NodeJS.Timeout | undefined;
    private _filterState: FilterState | null = null; // Store current filter state to inherit in diff view
    private static _isProgrammaticSelection: boolean = false; // Flag to prevent circular highlighting
    private _context: vscode.ExtensionContext | undefined;

    public static isProgrammaticSelection(): boolean {
        return AssemblyViewerPanel._isProgrammaticSelection;
    }

    public static createOrShow(
        extensionUri: vscode.Uri,
        compiler: TritonCompiler,
        document: vscode.TextDocument,
        highlightManager: HighlightManager,
        sourceLine?: number,
        context?: vscode.ExtensionContext
    ) {
        const column = vscode.ViewColumn.Beside;

        if (AssemblyViewerPanel.currentPanel) {
            AssemblyViewerPanel.currentPanel._panel.reveal(column);
            AssemblyViewerPanel.currentPanel.updateAssembly(document, sourceLine);
            AssemblyViewerPanel.currentPanel._context = context;
            // Save state
            if (context) {
                context.workspaceState.update('assemblyViewerState', {
                    documentUri: document.uri.toString(),
                    line: sourceLine || 0
                });
            }
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'trident',
            'Trident Assembly',
            column,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );

        // Set the icon for the webview panel
        panel.iconPath = vscode.Uri.joinPath(extensionUri, 'resources', 'icon.png');

        AssemblyViewerPanel.currentPanel = new AssemblyViewerPanel(
            panel,
            extensionUri,
            compiler,
            document,
            highlightManager,
            context
        );

        // Save state
        if (context) {
            context.workspaceState.update('assemblyViewerState', {
                documentUri: document.uri.toString(),
                line: sourceLine || 0
            });
        }
    }

    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        compiler: TritonCompiler,
        document: vscode.TextDocument,
        highlightManager: HighlightManager,
        context?: vscode.ExtensionContext
    ) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._compiler = compiler;
        this._highlightManager = highlightManager;
        this._currentDocument = document;
        this._context = context;

        // Initialize the list of open Triton files
        logger.log('Initializing assembly viewer panel');
        this.updateOpenTritonFiles();
        logger.log(`Found ${this._allOpenTritonFiles.size} open Triton files: ${Array.from(this._allOpenTritonFiles).map(f => path.basename(f)).join(', ')}`);

        this.updateAssembly(document);

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        // Listen for theme changes and refresh the view (debounced)
        vscode.window.onDidChangeActiveColorTheme(async (theme) => {
            logger.log(`Theme preview: ${theme.kind} (${vscode.ColorThemeKind[theme.kind]})`);

            // Clear any existing timeout
            if (this._themeChangeTimeout) {
                clearTimeout(this._themeChangeTimeout);
            }

            // Wait 800ms after the last theme change before refreshing
            // This prevents rebuilding while user is browsing themes with arrow keys
            this._themeChangeTimeout = setTimeout(async () => {
                logger.log(`Theme settled, refreshing assembly view`);

                // Force a complete rebuild of the webview
                if (this._diffState) {
                    const html = await this.getDiffViewContent(this._diffState, this._filterState);
                    this._panel.webview.html = '';
                    setTimeout(() => {
                        this._panel.webview.html = html;
                    }, 50);
                } else if (this._currentDocument) {
                    const html = await this.getWebviewContent();
                    this._panel.webview.html = '';
                    setTimeout(() => {
                        this._panel.webview.html = html;
                    }, 50);
                }
            }, 800);
        }, null, this._disposables);

        // Listen for text document open/close to update the source file dropdown
        vscode.workspace.onDidOpenTextDocument(async (doc) => {
            logger.log(`Document opened: ${path.basename(doc.uri.fsPath)} (language: ${doc.languageId})`);
            // Give VS Code a moment to fully load the document
            setTimeout(() => {
                if (doc.languageId === 'python') {
                    try {
                        const content = doc.getText();
                        if (content.includes('@triton.jit')) {
                            logger.log(`Found @triton.jit in ${path.basename(doc.uri.fsPath)}, adding to list`);
                            this._allOpenTritonFiles.add(doc.uri.fsPath);
                            this.refreshDropdown();
                        } else {
                            logger.log(`No @triton.jit found in ${path.basename(doc.uri.fsPath)}`);
                        }
                    } catch (e) {
                        logger.error(`Error reading document ${path.basename(doc.uri.fsPath)}`, e);
                    }
                }
            }, 100);
        }, null, this._disposables);

        vscode.workspace.onDidCloseTextDocument((doc) => {
            logger.log(`Document closed: ${path.basename(doc.uri.fsPath)}`);
            if (this._allOpenTritonFiles.has(doc.uri.fsPath)) {
                logger.log(`Removing ${path.basename(doc.uri.fsPath)} from list`);
                this._allOpenTritonFiles.delete(doc.uri.fsPath);
                this.refreshDropdown();
            }
        }, null, this._disposables);

        // Also listen for when the active editor changes (user switches tabs)
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor && editor.document.languageId === 'python') {
                logger.log(`Active editor changed to: ${path.basename(editor.document.uri.fsPath)}`);
                this.refreshDropdown();
            }
        }, null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'highlightSource':
                        await this.highlightSourceLine(message.line, message.kernelIndex);
                        break;
                    case 'refresh':
                        await this.refreshCache();
                        break;
                    case 'clearCache':
                        await this.clearCache();
                        break;
                    case 'filterByFile':
                        await this.filterBySourceFile(message.sourceFile);
                        break;
                    case 'openSourceFile':
                        await this.openSourceFile(message.sourceFile);
                        break;
                    case 'jumpToKernel':
                        await this.jumpToKernel(message.kernelIndex);
                        break;
                    case 'compareKernels':
                        await this.showCompareKernelPicker();
                        break;
                    case 'highlightSourceFromDiff':
                        await this.highlightSourceLineFromDiff(message.line, message.side);
                        break;
                    case 'closeDiff':
                        await this.closeDiff();
                        break;
                    case 'openCacheFile':
                        await this.openCacheFile(message.filePath);
                        break;
                    case 'saveFilterState':
                        this._filterState = message.filterState;
                        break;
                    case 'setLabel':
                        await this.setKernelLabel(message.cachePath, message.label);
                        break;
                    case 'requestLabelEdit':
                        await this.requestLabelEdit(message.cachePath, message.currentLabel);
                        break;
                    case 'filterByIRType':
                        await this.filterByIRType(message.irType);
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    public getCurrentDocument(): vscode.TextDocument | undefined {
        return this._currentDocument;
    }

    public async highlightAssemblyForSourceLine(document: vscode.TextDocument, line: number) {
        // Set the document for highlighting (supports clicking on any source file)
        // Keep this document as the current one (don't restore) to maintain proper state
        this._currentDocument = document;

        // Detect which kernel the cursor is in
        const pythonSource = document.getText();
        const kernelName = this._compiler.detectKernelAtLine(pythonSource, line);

        if (kernelName) {
            // Find the kernel in the FILTERED assemblies (maintain current IR type)
            const filteredKernel = this._assemblies.find(asm => asm.kernelName === kernelName);

            if (filteredKernel) {
                const kernelIndex = this._allAssemblies.indexOf(filteredKernel);

                if (kernelIndex >= 0 && kernelIndex !== this._selectedKernelIndex) {
                    // Auto-switch to the kernel the cursor is in (within current IR type)
                    logger.log(`[Trident] Auto-switching to kernel: ${kernelName} (${this._selectedIRType})`);
                    await this.jumpToKernel(kernelIndex);
                    // Don't return here - continue to highlight the line
                }
            }
        }

        this.highlightAssemblyFromSource(line);
    }

    public async updateAssembly(document: vscode.TextDocument, highlightSourceLine?: number) {
        this._currentDocument = document;

        try {
            // Step 1: Extract kernel names ONLY from the current document (not all open files)
            const kernelNames = new Set<string>();
            const content = document.getText();
            if (content.includes('@triton.jit')) {
                const names = this._compiler.extractKernelNames(content);
                names.forEach(name => kernelNames.add(name));
                logger.log(`[Trident] Found ${names.length} kernels in ${path.basename(document.uri.fsPath)}: ${names.join(', ')}`);
            }

            if (kernelNames.size === 0) {
                logger.log(`[Trident] No kernel names found in ${path.basename(document.uri.fsPath)}`);
                this._panel.webview.html = this.getNoAssemblyHtml();
                return;
            }

            logger.log(`[Trident] Searching cache for ${kernelNames.size} kernels from ${path.basename(document.uri.fsPath)}: ${Array.from(kernelNames).join(', ')}`);

            // Step 2: Search cache and match by kernel name
            const workspaceFolders = vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath) || [];
            this._allAssemblies = await this._compiler.findCachedAssemblies(kernelNames, workspaceFolders);

            logger.log(`Loaded ${this._allAssemblies.length} assemblies for ${path.basename(document.uri.fsPath)}`);
            this._allAssemblies.forEach((asm, i) => {
                logger.log(`  [${i}] ${asm.kernelName} (${this.formatTimestamp(asm.timestamp)})`);
            });

            // Filter by IR type (maintains user's selection)
            const filteredAssemblies = this._allAssemblies.filter(asm => asm.irType === this._selectedIRType);

            // Show ALL kernels (not just the first one)
            this._assemblies = filteredAssemblies;

            // Auto-select the kernel at the cursor position (if highlightSourceLine is provided)
            if (highlightSourceLine !== undefined) {
                const pythonSource = document.getText();
                const kernelName = this._compiler.detectKernelAtLine(pythonSource, highlightSourceLine);
                if (kernelName) {
                    // Find the kernel in FILTERED assemblies (respect IR type selection)
                    const filteredKernel = this._assemblies.find(asm => asm.kernelName === kernelName);
                    if (filteredKernel) {
                        const kernelIndex = this._allAssemblies.indexOf(filteredKernel);
                        this._selectedKernelIndex = kernelIndex;
                        logger.log(`[Trident] Auto-selected kernel at cursor: ${kernelName} (${this._selectedIRType}, index ${kernelIndex})`);
                    } else {
                        // Kernel not found in current IR type, select first filtered kernel
                        this._selectedKernelIndex = this._assemblies.length > 0 ? this._allAssemblies.indexOf(this._assemblies[0]) : 0;
                    }
                } else {
                    // No kernel detected at cursor, select first filtered kernel
                    this._selectedKernelIndex = this._assemblies.length > 0 ? this._allAssemblies.indexOf(this._assemblies[0]) : 0;
                }
            } else {
                // No source line specified, select first filtered kernel
                this._selectedKernelIndex = this._assemblies.length > 0 ? this._allAssemblies.indexOf(this._assemblies[0]) : 0;
            }

            if (this._assemblies.length === 0) {
                this._panel.webview.html = this.getNoAssemblyHtml();
            } else {
                // Build source-to-assembly line mapping from .loc directives
                this.buildLineMapping();

                this._panel.webview.html = await this.getWebviewContent();

                // Refresh dropdown to include all files from cache (not just open ones)
                await this.refreshDropdown();

                // If a source line was specified, highlight corresponding assembly
                if (highlightSourceLine !== undefined) {
                    setTimeout(() => {
                        this.highlightAssemblyFromSource(highlightSourceLine);
                    }, 100);
                }
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load assembly: ${error}`);
            this._panel.webview.html = this.getErrorHtml(String(error));
        }
    }

    /**
     * Build mapping from source lines to assembly lines using .loc directives
     * Maps for ALL source files (including inlined helpers)
     * Key format: "normalized_filepath:line"
     */
    private buildLineMapping() {
        // Use modular mapping builder to preprocess all mappings
        this._lineMapping = this._mappingBuilder.buildMapping(this._allAssemblies);
    }

    /**
     * Highlight assembly lines corresponding to a source line
     * Uses file path to support cross-file highlighting (e.g., inlined helpers)
     */
    private highlightAssemblyFromSource(sourceLine: number) {
        if (!this._currentDocument) {
            return;
        }

        // Create key with current document's file path
        const normalizedPath = path.normalize(this._currentDocument.uri.fsPath);
        const key = `${normalizedPath}:${sourceLine}`;

        // Collect all assembly lines to highlight
        const allAsmLines = new Set<number>();

        // First, add direct mappings for this source line
        const directAsmLines = this._lineMapping.sourceToAsm.get(key);
        if (directAsmLines) {
            directAsmLines.forEach(line => allAsmLines.add(line));
        }

        // Check if this is a callsite (inlined function call)
        // Look in TTIR/TTGIR assemblies for callsite info
        // Also check nearby lines (±5 lines) to handle multi-line function calls
        const selectedIdx = this._selectedKernelIndex ?? 0;
        if (selectedIdx >= 0 && selectedIdx < this._allAssemblies.length) {
            // Find the TTIR assembly for this kernel (same kernel name, different IR type)
            const selectedKernel = this._allAssemblies[selectedIdx];
            const ttirAssembly = this._allAssemblies.find(asm =>
                asm.kernelName === selectedKernel.kernelName &&
                (asm.irType === 'ttir' || asm.irType === 'ttgir')
            );

            if (ttirAssembly?.callsites) {
                // Check current line and nearby lines for callsites
                const linesToCheck = [];
                for (let offset = -5; offset <= 5; offset++) {
                    const checkLine = sourceLine + offset;
                    if (checkLine >= 0) {
                        linesToCheck.push(`${normalizedPath}:${checkLine}`);
                    }
                }

                let foundCallsite = false;
                for (const checkKey of linesToCheck) {
                    const inlinedLocations = ttirAssembly.callsites.callsiteMap.get(checkKey);
                    if (inlinedLocations && inlinedLocations.length > 0) {
                        if (!foundCallsite) {
                            logger.log(`[Trident] Found callsite at ${path.basename(checkKey)} with ${inlinedLocations.length} inlined locations`);
                            foundCallsite = true;
                        }

                        // For each inlined location, find corresponding assembly lines
                        for (const inlinedKey of inlinedLocations) {
                            const inlinedAsmLines = this._lineMapping.sourceToAsm.get(inlinedKey);
                            if (inlinedAsmLines) {
                                inlinedAsmLines.forEach(line => allAsmLines.add(line));
                                logger.log(`  + ${inlinedAsmLines.length} lines from ${path.basename(inlinedKey)}`);
                            }
                        }
                    }
                }
            }
        }

        if (allAsmLines.size > 0) {
            // Calculate the global line range for the selected kernel
            let globalStartLine = 0;
            for (let i = 0; i < selectedIdx; i++) {
                globalStartLine += this._allAssemblies[i].assembly.split('\n').length;
            }
            const selectedKernelLines = this._allAssemblies[selectedIdx].assembly.split('\n').length;
            const globalEndLine = globalStartLine + selectedKernelLines;

            // Filter and convert to local line numbers
            const localLines = Array.from(allAsmLines)
                .filter(line => line >= globalStartLine && line < globalEndLine)
                .map(line => line - globalStartLine);

            if (localLines.length > 0) {
                logger.log(`Highlighting ${localLines.length} assembly lines for ${path.basename(normalizedPath)}:${sourceLine + 1} in kernel ${this._allAssemblies[selectedIdx].kernelName}`);
                this._panel.webview.postMessage({
                    command: 'highlightAssembly',
                    lines: localLines
                });
            } else {
                logger.log(`No assembly lines for ${path.basename(normalizedPath)}:${sourceLine + 1} in selected kernel ${this._allAssemblies[selectedIdx].kernelName}`);
            }
        }
    }


    private async refreshCache() {
        if (this._currentDocument) {
            await this.updateAssembly(this._currentDocument);
        }
    }

    private async clearCache() {
        const cachePath = this._compiler.getCachePath();
        const result = await vscode.window.showWarningMessage(
            `Are you sure you want to delete the Triton cache?\n\nCache location: ${cachePath}\n\nThis action cannot be undone.`,
            { modal: true },
            'Delete Cache'
        );

        if (result === 'Delete Cache') {
            try {
                await this._compiler.clearCache();
                vscode.window.showInformationMessage('Triton cache cleared successfully');

                // Refresh the view to show empty state
                if (this._currentDocument) {
                    await this.updateAssembly(this._currentDocument);
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to clear cache: ${error}`);
            }
        }
    }

    /**
     * Get custom label for a kernel assembly (by cache path)
     */
    private getKernelLabel(cachePath: string): string | undefined {
        if (!this._context) return undefined;
        const labels = this._context.workspaceState.get<Record<string, string>>('kernelLabels', {});
        return labels[cachePath];
    }

    /**
     * Set custom label for a kernel assembly
     */
    private async setKernelLabel(cachePath: string, label: string | undefined): Promise<void> {
        if (!this._context) return;

        const labels = this._context.workspaceState.get<Record<string, string>>('kernelLabels', {});

        if (label && label.trim()) {
            labels[cachePath] = label.trim();
        } else {
            delete labels[cachePath];
        }

        await this._context.workspaceState.update('kernelLabels', labels);

        // Refresh the view to show updated label
        if (this._currentDocument) {
            await this.updateAssembly(this._currentDocument);
        }
    }

    /**
     * Get display text for a kernel (timestamp + custom label if set)
     */
    private getKernelDisplayText(asm: KernelAssembly): string {
        const timestamp = this.formatTimestamp(asm.timestamp);
        const customLabel = this.getKernelLabel(asm.cachePath);
        if (customLabel) {
            return `${timestamp} - ${customLabel}`;
        }
        return timestamp;
    }

    /**
     * Request label edit - show input dialog and set label
     */
    private async requestLabelEdit(cachePath: string, currentLabel: string | undefined): Promise<void> {
        const label = await vscode.window.showInputBox({
            prompt: 'Enter a custom label for this kernel version',
            placeHolder: 'e.g., "Before optimization", "Version 2.0"',
            value: currentLabel || '',
            validateInput: (value) => {
                // Allow empty to remove label
                return null;
            }
        });

        if (label !== undefined) {
            // If user pressed Enter with empty value, remove the label
            await this.setKernelLabel(cachePath, label || undefined);
        }
    }

    private updateOpenTritonFiles() {
        // Scan all currently open documents for Triton files
        logger.log('Scanning all open documents for Triton files...');
        this._allOpenTritonFiles.clear();
        for (const document of vscode.workspace.textDocuments) {
            if (document.languageId === 'python' && !document.isClosed) {
                try {
                    const content = document.getText();
                    if (content.includes('@triton.jit')) {
                        logger.log(`  Found Triton file: ${path.basename(document.uri.fsPath)}`);
                        this._allOpenTritonFiles.add(document.uri.fsPath);
                    }
                } catch (e) {
                    logger.error(`  Error reading ${path.basename(document.uri.fsPath)}`, e);
                }
            }
        }
        logger.log(`Total Triton files in cache: ${this._allOpenTritonFiles.size}`);
    }

    private async refreshDropdown() {
        logger.log('Refreshing dropdown...');
        logger.log(`Before scan: ${this._allOpenTritonFiles.size} files in cache`);

        // Re-scan all open files to ensure we're up to date
        this.updateOpenTritonFiles();

        // Also get all source files from the cache (not just open files)
        const allSourceFiles = new Set<string>(this._allOpenTritonFiles);

        try {
            // Extract kernel names from all open Python files
            const kernelNames = new Set<string>();
            for (const doc of vscode.workspace.textDocuments) {
                if (doc.languageId === 'python' && !doc.isClosed) {
                    const content = doc.getText();
                    if (content.includes('@triton.jit')) {
                        const names = this._compiler.extractKernelNames(content);
                        names.forEach(name => kernelNames.add(name));
                    }
                }
            }

            // Get ALL assemblies from cache for these kernels
            const workspaceFolders = vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath) || [];
            const allAssemblies = kernelNames.size > 0 ? await this._compiler.findCachedAssemblies(kernelNames, workspaceFolders) : [];

            // Extract all unique source files from all assemblies
            for (const asm of allAssemblies) {
                for (const sourceFile of asm.sourceFiles) {
                    allSourceFiles.add(sourceFile);
                }
            }

            logger.log(`Found ${allAssemblies.length} total assemblies with ${allSourceFiles.size} unique source files`);
        } catch (error) {
            logger.error('Error scanning cache for source files', error);
        }

        logger.log(`After scan: ${allSourceFiles.size} files: ${Array.from(allSourceFiles).map(f => path.basename(f)).join(', ')}`);

        // Send updated file list to webview via postMessage (lightweight update)
        const fileList = Array.from(allSourceFiles).map(file => ({
            path: file,
            basename: path.basename(file)
        }));

        this._panel.webview.postMessage({
            command: 'updateFileDropdown',
            files: fileList,
            selectedFile: this._selectedSourceFile
        });

        logger.log('Sent file dropdown update via postMessage');
    }

    private async filterByIRType(irType: string) {
        logger.log(`Filtering by IR type: ${irType}`);
        this._selectedIRType = irType as IRType;

        // Just filter the displayed assemblies, don't rebuild line mapping
        // (line mapping is already built for all assemblies)
        const filteredAssemblies = this._allAssemblies.filter(asm => asm.irType === this._selectedIRType);

        // Update displayed assemblies (show ALL kernels)
        this._assemblies = filteredAssemblies;

        // Set selected kernel index to first filtered assembly
        if (this._assemblies.length > 0) {
            this._selectedKernelIndex = this._allAssemblies.indexOf(this._assemblies[0]);
        } else {
            this._selectedKernelIndex = 0;
        }

        // Clear highlights when changing IR type filter
        this._highlightManager.clearHighlights();

        logger.log(`Filtered to ${this._assemblies.length} assemblies of type ${irType}`);

        // Update the webview without rebuilding line mapping
        if (this._assemblies.length === 0) {
            this._panel.webview.html = this.getNoAssemblyHtml();
        } else {
            this._panel.webview.html = await this.getWebviewContent();
        }
    }

    private async filterBySourceFile(sourceFile: string) {
        logger.log(`Filtering by source file: ${sourceFile ? path.basename(sourceFile) : 'current document'}`);

        // If empty, reload with current document
        if (!sourceFile && this._currentDocument) {
            this._selectedSourceFile = undefined;
            await this.updateAssembly(this._currentDocument);
            return;
        }

        this._selectedSourceFile = sourceFile;

        // Extract kernel names ONLY from the selected source file
        const kernelNames = new Set<string>();
        const targetDoc = vscode.workspace.textDocuments.find(doc =>
            doc.uri.fsPath === sourceFile && doc.languageId === 'python' && !doc.isClosed
        );

        if (targetDoc) {
            const content = targetDoc.getText();
            if (content.includes('@triton.jit')) {
                const names = this._compiler.extractKernelNames(content);
                names.forEach(name => kernelNames.add(name));
                logger.log(`Found ${names.length} kernels in ${path.basename(sourceFile)}: ${names.join(', ')}`);
            }
        }

        if (kernelNames.size === 0) {
            logger.log(`No kernels found in ${path.basename(sourceFile)}`);
            vscode.window.showWarningMessage(`No Triton kernels found in ${path.basename(sourceFile)}`);
            // Don't switch - stay on current view
            return;
        }

        // Load assemblies ONLY for kernels in the selected file
        const workspaceFolders = vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath) || [];
        this._allAssemblies = await this._compiler.findCachedAssemblies(kernelNames, workspaceFolders);

        logger.log(`Loaded ${this._allAssemblies.length} assemblies for kernels: ${Array.from(kernelNames).join(', ')}`);

        // Filter by the currently selected IR type (maintain user's selection)
        const filteredAssemblies = this._allAssemblies.filter(asm => asm.irType === this._selectedIRType);
        this._assemblies = filteredAssemblies;

        logger.log(`Showing ${this._assemblies.length} assemblies after IR filter (${this._selectedIRType})`);
        if (this._assemblies.length > 0) {
            logger.log(`Available kernels: ${this._assemblies.map(a => a.kernelName).join(', ')}`);
        }

        if (this._assemblies.length === 0) {
            vscode.window.showWarningMessage(`No ${this._selectedIRType.toUpperCase()} assemblies found for ${path.basename(sourceFile)}. Try a different IR type or run the file.`);
            // Don't switch - stay on current view
            return;
        }

        // Default to showing the first kernel
        this._selectedKernelIndex = this._allAssemblies.indexOf(this._assemblies[0]);

        // Build line mapping and update view
        this.buildLineMapping();
        this._panel.webview.html = await this.getWebviewContent();

        // Find the first line in assembly from this source file and jump to it
        setTimeout(async () => {
            const firstLine = await this.findFirstLineFromFile(sourceFile);
            if (firstLine !== undefined) {
                // Open the source file at the first line
                await this.openSourceFile(sourceFile, firstLine.sourceLine);

                // Scroll assembly to that location
                this._panel.webview.postMessage({
                    command: 'scrollToLine',
                    line: firstLine.asmLine
                });
            } else {
                // Just open the file at the top
                await this.openSourceFile(sourceFile);
            }
        }, 100);
    }

    private async findFirstLineFromFile(sourceFile: string): Promise<{ sourceLine: number, asmLine: number } | undefined> {
        // Parse all assemblies to find the first .loc directive referencing this file
        for (const asm of this._assemblies) {
            const lines = asm.assembly.split('\n');
            const fileMap = new Map<number, string>();

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];

                // Parse .file directive
                const format2Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"/);
                if (format2Match) {
                    const fileId = parseInt(format2Match[1]);
                    const directory = format2Match[2].replace(/^;/, '');
                    const filename = format2Match[3];
                    fileMap.set(fileId, path.join(directory, filename));
                    continue;
                }

                const format1Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"/);
                if (format1Match) {
                    const fileId = parseInt(format1Match[1]);
                    fileMap.set(fileId, format1Match[2]);
                    continue;
                }

                // Find first .loc directive for this file
                const locMatch = line.match(/\.loc\s+(\d+)\s+(\d+)\s+\d+/);
                if (locMatch) {
                    const fileId = parseInt(locMatch[1]);
                    const sourceLine = parseInt(locMatch[2]) - 1;

                    if (fileMap.has(fileId) && fileMap.get(fileId) === sourceFile) {
                        return { sourceLine, asmLine: i };
                    }
                }
            }
        }

        return undefined;
    }

    private async openSourceFile(sourceFile: string, line?: number, preserveFocus: boolean = false) {
        if (!sourceFile) {
            return;
        }

        try {
            const uri = vscode.Uri.file(sourceFile);
            const document = await vscode.workspace.openTextDocument(uri);
            const editor = await vscode.window.showTextDocument(document, {
                viewColumn: vscode.ViewColumn.One,
                preserveFocus: preserveFocus,  // Configurable based on context
                preview: false
            });

            // Jump to specific line if provided
            if (line !== undefined && line >= 0 && line < document.lineCount) {
                // Set flag to prevent circular highlighting
                AssemblyViewerPanel._isProgrammaticSelection = true;

                const range = document.lineAt(line).range;
                editor.selection = new vscode.Selection(range.start, range.start);
                editor.revealRange(range, vscode.TextEditorRevealType.InCenter);

                // Clear flag after a short delay
                setTimeout(() => {
                    AssemblyViewerPanel._isProgrammaticSelection = false;
                }, 200);
            }
        } catch (error) {
            vscode.window.showWarningMessage(`Could not open file: ${path.basename(sourceFile)}`);
            console.error('Error opening source file:', error);
        }
    }

    private async openCacheFile(filePath: string) {
        if (!filePath) {
            return;
        }

        try {
            const uri = vscode.Uri.file(filePath);
            await vscode.window.showTextDocument(uri, {
                viewColumn: vscode.ViewColumn.Beside,
                preserveFocus: true,
                preview: false
            });
        } catch (error) {
            vscode.window.showWarningMessage(`Could not open cache file: ${path.basename(filePath)}`);
            console.error('Error opening cache file:', error);
        }
    }

    private async jumpToKernel(kernelIndex: number) {
        logger.log(`Jump to kernel: index=${kernelIndex}, total kernels=${this._allAssemblies.length}`);

        // kernelIndex is relative to _allAssemblies (from the dropdown)
        if (kernelIndex < 0 || kernelIndex >= this._allAssemblies.length) {
            logger.log(`Invalid kernel index: ${kernelIndex}`);
            return;
        }

        // Update selected kernel index (don't filter _assemblies, keep all of them in the dropdown)
        this._selectedKernelIndex = kernelIndex;

        const selectedKernel = this._allAssemblies[kernelIndex];
        logger.log(`Selected kernel: ${selectedKernel.kernelName} [${selectedKernel.irType}]`);

        // Clear any stale source code highlights when switching kernels
        this._highlightManager.clearHighlights();

        // Rebuild the view to show the selected kernel (but keep all kernels in dropdown)
        this.buildLineMapping();
        this._panel.webview.html = await this.getWebviewContent();

        // Scroll to top of assembly after rebuild
        setTimeout(() => {
            this._panel.webview.postMessage({
                command: 'scrollToTop'
            });
        }, 100);

        const kernel = this._allAssemblies[kernelIndex];
        const lines = kernel.assembly.split('\n');

        // Find the first .loc directive in this kernel to get the source file and line
        let sourceFile: string | undefined;
        let sourceLine = -1;
        const fileMap = new Map<number, string>();

        for (const line of lines) {
            // Parse .file directive
            const format2Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"/);
            if (format2Match) {
                const fileId = parseInt(format2Match[1]);
                const directory = format2Match[2].replace(/^;/, '');
                const filename = format2Match[3];
                fileMap.set(fileId, path.join(directory, filename));
                continue;
            }

            const format1Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"/);
            if (format1Match) {
                const fileId = parseInt(format1Match[1]);
                fileMap.set(fileId, format1Match[2]);
                continue;
            }

            // Find first .loc directive
            const locMatch = line.match(/\.loc\s+(\d+)\s+(\d+)\s+\d+/);
            if (locMatch) {
                const fileId = parseInt(locMatch[1]);
                sourceLine = parseInt(locMatch[2]) - 1; // Convert to 0-indexed

                if (fileMap.has(fileId)) {
                    sourceFile = fileMap.get(fileId);
                }

                // Found the first one, break
                if (sourceFile && sourceLine >= 0) {
                    break;
                }
            }
        }

        // Open source file at the line
        if (sourceFile && sourceLine >= 0) {
            logger.log(`Jumping to first line: ${path.basename(sourceFile)}:${sourceLine + 1}`);
            await this.openSourceFile(sourceFile, sourceLine);
        } else {
            logger.log('No source line found for this kernel');
        }
    }

    private async showCompareKernelPicker() {
        if (this._selectedKernelIndex === undefined || this._allAssemblies.length < 2) {
            vscode.window.showInformationMessage('Need at least 2 kernel versions to compare');
            return;
        }

        const currentKernel = this._allAssemblies[this._selectedKernelIndex];
        const currentIRType = currentKernel.irType;

        // Build quick pick items for other kernels of the same IR type
        const items = this._allAssemblies
            .map((asm, index) => ({
                label: asm.kernelName,
                description: this.getKernelDisplayText(asm),
                detail: asm.cachePath,
                index: index,
                irType: asm.irType
            }))
            .filter(item => item.index !== this._selectedKernelIndex && item.irType === currentIRType);

        if (items.length === 0) {
            vscode.window.showInformationMessage(`No other ${currentIRType.toUpperCase()} kernels found to compare`);
            return;
        }

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: `Compare ${currentKernel.kernelName} (${this.getKernelDisplayText(currentKernel)}) with...`
        });

        if (selected) {
            await this.compareKernels(this._selectedKernelIndex, selected.index);
        }
    }

    private async compareKernels(index1: number, index2: number) {
        const kernel1 = this._allAssemblies[index1];
        const kernel2 = this._allAssemblies[index2];

        logger.log(`Comparing kernels: ${kernel1.kernelName} vs ${kernel2.kernelName}`);

        try {
            // Store both kernels for diff view
            this._diffState = {
                kernel1,
                kernel2,
                index1,
                index2
            };

            // Show custom diff view in the webview (pass current filter state)
            this._panel.webview.html = await this.getDiffViewContent(this._diffState, this._filterState);

            logger.log('Opened custom diff view');
        } catch (error) {
            logger.error('Error comparing kernels:', error);
            vscode.window.showErrorMessage(`Failed to compare kernels: ${error}`);
        }
    }

    private async closeDiff() {
        logger.log('Closing diff view, returning to regular view');
        this._diffState = undefined;

        // Rebuild the regular assembly view
        if (this._currentDocument) {
            this._panel.webview.html = await this.getWebviewContent();

            // Restore filter state after rebuilding
            if (this._filterState) {
                logger.log(`Restoring filter state after closing diff: ${JSON.stringify(this._filterState)}`);
                setTimeout(() => {
                    this._panel.webview.postMessage({
                        command: 'restoreFilters',
                        filterState: this._filterState
                    });
                }, 200); // Wait for webview to load
            }
        }
    }

    private async highlightSourceLineFromDiff(asmLine: number, side: string) {
        if (!this._diffState) {
            return;
        }

        // Get the kernel based on which side was clicked
        const kernel = side === 'left' ? this._diffState.kernel1 : this._diffState.kernel2;
        const lines = kernel.assembly.split('\n');

        // Find which source line(s) and file correspond to this assembly line
        let sourceLine = -1;
        let sourceFile: string | undefined;

        if (asmLine < lines.length) {
            // Walk backwards to find the most recent .file and .loc directives
            let currentFileId = -1;
            const fileMap = new Map<number, string>();

            for (let i = 0; i <= asmLine; i++) {
                const line = lines[i];

                // Parse .file directive
                const format2Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"/);
                if (format2Match) {
                    const fileId = parseInt(format2Match[1]);
                    const directory = format2Match[2].replace(/^;/, '');
                    const filename = format2Match[3];
                    fileMap.set(fileId, path.join(directory, filename));
                    continue;
                }

                const format1Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"/);
                if (format1Match) {
                    const fileId = parseInt(format1Match[1]);
                    fileMap.set(fileId, format1Match[2]);
                    continue;
                }

                // Parse .loc directive
                const locMatch = line.match(/\.loc\s+(\d+)\s+(\d+)\s+\d+/);
                if (locMatch && i <= asmLine) {
                    currentFileId = parseInt(locMatch[1]);
                    sourceLine = parseInt(locMatch[2]) - 1;
                }
            }

            if (currentFileId >= 0 && fileMap.has(currentFileId)) {
                sourceFile = fileMap.get(currentFileId);
            }
        }

        if (sourceLine >= 0 && sourceFile) {
            await this.openSourceFile(sourceFile, sourceLine, true);  // Preserve focus when clicking diff
        }
    }

    private getFilterClasses(line: string): string {
        const trimmedLine = line.trim();
        const isDirective = trimmedLine.startsWith('.') ||
            trimmedLine.startsWith('-') ||
            trimmedLine.startsWith('amdhsa') ||
            trimmedLine === '---' ||
            trimmedLine === '...';
        const isComment = trimmedLine.startsWith('#') || trimmedLine.startsWith(';') || trimmedLine.startsWith('//');
        const isEmpty = trimmedLine.length === 0;

        const filterClasses = [];
        if (isDirective) {
            filterClasses.push('filter-directive');
        }
        if (isComment) {
            filterClasses.push('filter-comment');
        }
        if (isEmpty) {
            filterClasses.push('filter-empty');
        }

        // Extract instruction name and add as class
        if (!isDirective && !isComment && !isEmpty) {
            const instrMatch = trimmedLine.match(/^\s*(\w+)/);
            if (instrMatch) {
                const instr = instrMatch[1].toLowerCase();

                // Filter out labels and amdhsa/metadata keywords
                const isLabel = trimmedLine.trim().endsWith(':');
                const isAmdHsa = instr.startsWith('amdhsa') || instr === 'end_amd_kernel_code_t';
                const isMetadata = instr === 'amd_kernel_code_t' || instr === 'kernel_code_entry_byte_offset' ||
                    instr === 'kernel_code_prefetch_byte_size' || instr === 'max_scratch_backing_memory_byte_size';

                if (!isLabel && !isAmdHsa && !isMetadata) {
                    filterClasses.push('instr-' + instr);
                }
            }
        }

        return filterClasses.join(' ');
    }

    private async getDiffViewContent(diffState: { kernel1: KernelAssembly, kernel2: KernelAssembly, index1: number, index2: number }, filterState: FilterState | null): Promise<string> {
        const { kernel1, kernel2 } = diffState;
        const inheritedFilterState = filterState ? JSON.stringify(filterState) : 'null';

        // Determine labels based on timestamps (don't swap kernels - user selected order)
        const isKernel1Older = kernel1.timestamp.getTime() < kernel2.timestamp.getTime();
        const leftLabel = isKernel1Older ? 'Old' : 'New';
        const rightLabel = isKernel1Older ? 'New' : 'Old';

        // Simple line-by-line diff
        const lines1 = kernel1.assembly.split('\n');
        const lines2 = kernel2.assembly.split('\n');

        const maxLines = Math.max(lines1.length, lines2.length);

        let leftHtml = '';
        let rightHtml = '';

        for (let i = 0; i < maxLines; i++) {
            const line1 = lines1[i] || '';
            const line2 = lines2[i] || '';

            // Simple comparison - mark as changed if different
            const isDifferent = line1 !== line2;
            const diffClass = isDifferent ? 'diff-changed' : '';

            const lineNum = (i + 1).toString().padStart(4, ' ');

            // Add filter classes for both sides
            const filterClasses1 = this.getFilterClasses(line1);
            const filterClasses2 = this.getFilterClasses(line2);

            leftHtml += `<div class="asm-line clickable ${diffClass} ${filterClasses1}" data-line="${i}" data-side="left"><span class="line-number">${lineNum}</span><span class="asm-content">${this.highlightGCNSyntax(line1 || ' ')}</span></div>\n`;
            rightHtml += `<div class="asm-line clickable ${diffClass} ${filterClasses2}" data-line="${i}" data-side="right"><span class="line-number">${lineNum}</span><span class="asm-content">${this.highlightGCNSyntax(line2 || ' ')}</span></div>\n`;
        }

        // Use custom labels if available, otherwise timestamps
        const displayText1 = this.getKernelDisplayText(kernel1);
        const displayText2 = this.getKernelDisplayText(kernel2);

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${this._panel.webview.cspSource} 'unsafe-inline' https:; script-src 'unsafe-inline'; font-src ${this._panel.webview.cspSource} https:;">
    <link href="https://unpkg.com/@vscode/codicons@latest/dist/codicon.css" rel="stylesheet" />
    <title>Compare: ${kernel1.kernelName}</title>
    <style>
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            padding: 0;
            margin: 0;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }

        .toolbar {
            position: sticky;
            top: 0;
            background-color: var(--vscode-editorWidget-background);
            border-bottom: 1px solid var(--vscode-widget-border);
            padding: 8px 16px;
            display: flex;
            gap: 10px;
            align-items: center;
            z-index: 100;
        }

        .btn {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 13px;
        }

        .btn:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        .icon-btn {
            background-color: transparent;
            color: var(--vscode-foreground);
            border: none;
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 16px;
        }

        .icon-btn:hover {
            background-color: var(--vscode-toolbar-hoverBackground);
        }

        .edit-label-btn {
            margin-right: 4px;
            padding: 2px 4px;
            opacity: 0.6;
            vertical-align: middle;
        }

        .edit-label-btn:hover {
            opacity: 1;
        }

        .icon-btn .codicon {
            font-size: 16px;
        }

        .filter-menu {
            position: fixed;
            background-color: var(--vscode-dropdown-background);
            border: 1px solid var(--vscode-dropdown-border);
            border-radius: 2px;
            padding: 8px;
            z-index: 1000;
            min-width: 300px;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .filter-section {
            margin-bottom: 12px;
        }

        .filter-section:last-child {
            margin-bottom: 0;
        }

        .filter-section-title {
            font-weight: bold;
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 6px;
            padding: 0 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .filter-menu label {
            display: block;
            padding: 4px 0;
            cursor: pointer;
            color: var(--vscode-dropdown-foreground);
            user-select: none;
        }

        .filter-menu label:hover {
            background-color: var(--vscode-list-hoverBackground);
        }

        .filter-menu input[type="checkbox"] {
            margin-right: 8px;
            cursor: pointer;
        }

        .filter-search-container {
            padding: 4px 8px 8px 8px;
        }

        .filter-search-container input {
            width: 100%;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 12px;
            outline: none;
        }

        .filter-search-container input:focus {
            border-color: var(--vscode-focusBorder);
        }

        .filter-actions {
            display: flex;
            gap: 4px;
            padding: 0 8px 8px 8px;
        }

        .filter-action-btn {
            flex: 1;
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            border: none;
            padding: 4px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 2px;
        }

        .filter-action-btn:hover {
            background-color: var(--vscode-button-secondaryHoverBackground);
        }

        #diffInstructionList {
            max-height: 300px;
            overflow-y: auto;
        }

        #diffInstructionList label {
            display: block;
        }

        #diffInstructionList label.filter-hidden {
            display: none;
        }

        .asm-line.hidden {
            display: none;
        }

        .search-bar {
            position: sticky;
            top: 44px;
            background-color: var(--vscode-editorWidget-background);
            border-bottom: 1px solid var(--vscode-widget-border);
            padding: 6px 12px;
            display: flex;
            gap: 6px;
            align-items: center;
            z-index: 99;
        }

        .search-bar input {
            flex: 1;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 13px;
            outline: none;
        }

        .search-bar input:focus {
            border-color: var(--vscode-focusBorder);
        }

        .search-nav-btn, .search-close-btn {
            background-color: transparent;
            color: var(--vscode-foreground);
            border: none;
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 12px;
        }

        .search-nav-btn:hover, .search-close-btn:hover {
            background-color: var(--vscode-toolbar-hoverBackground);
        }

        .search-results {
            color: var(--vscode-descriptionForeground);
            font-size: 12px;
            white-space: nowrap;
        }

        .search-match {
            background-color: rgba(255, 230, 100, 0.35) !important;
            outline: 2px solid rgba(255, 200, 0, 0.6);
            outline-offset: -2px;
        }

        .search-match-current {
            background-color: rgba(255, 140, 0, 0.5) !important;
            outline: 2px solid rgba(255, 100, 0, 0.9);
            outline-offset: -2px;
        }

        .diff-container {
            display: flex;
            height: calc(100vh - 50px);
        }

        .diff-side {
            flex: 1;
            overflow-y: auto;
            overflow-x: auto;
            border-right: 1px solid var(--vscode-widget-border);
        }

        .diff-side:last-child {
            border-right: none;
        }

        .diff-header {
            position: sticky;
            top: 0;
            background-color: var(--vscode-editorGroupHeader-tabsBackground);
            padding: 8px 12px;
            font-weight: bold;
            border-bottom: 1px solid var(--vscode-widget-border);
            z-index: 50;
        }

        .asm-line {
            padding: 2px 0;
            font-size: 13px;
            line-height: 1.6;
            cursor: pointer;
            display: block;
            white-space: pre;
        }

        .line-number {
            color: var(--vscode-editorLineNumber-foreground);
            padding: 0 12px 0 4px;
            user-select: none;
            font-family: monospace;
            display: inline-block;
            width: 60px;
            text-align: right;
            white-space: pre;
        }

        .asm-content {
            white-space: pre;
        }

        .asm-line:hover {
            background-color: var(--vscode-list-hoverBackground);
        }

        .asm-line.diff-changed {
            background-color: var(--vscode-diffEditor-insertedTextBackground, rgba(155, 185, 85, 0.2));
        }

        .diff-side:first-child .asm-line.diff-changed {
            background-color: var(--vscode-diffEditor-removedTextBackground, rgba(255, 0, 0, 0.2));
        }

        .asm-line.highlighted {
            background-color: var(--vscode-editor-selectionBackground);
        }

        /* GCN Assembly Syntax Highlighting */
        .comment {
            color: var(--vscode-descriptionForeground);
            font-style: italic;
        }
        .directive {
            color: var(--vscode-textLink-foreground);
            font-weight: bold;
        }
        .instruction {
            color: var(--vscode-textPreformat-foreground);
            font-weight: 500;
        }
        .register {
            color: var(--vscode-editor-foreground);
            opacity: 0.9;
        }
        .immediate {
            color: var(--vscode-charts-green);
        }
        .label {
            color: var(--vscode-textLink-activeForeground);
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <button class="btn" onclick="closeDiff()">← Back to Assembly</button>
        <span style="flex: 1; text-align: center; font-weight: bold;">
            Comparing: ${kernel1.kernelName}
        </span>
        <button class="icon-btn" id="diffSearchBtn" onclick="toggleDiffSearch()" title="Search in diff (Ctrl+F)">
            <i class="codicon codicon-search"></i>
        </button>
        <button class="icon-btn" id="diffFilterBtn" onclick="toggleDiffFilterMenu(event)" title="Filter assembly lines - hide directives, comments, instructions">
            <i class="codicon codicon-filter"></i>
        </button>
        <div class="filter-menu" id="diffFilterMenu" style="display: none;">
            <div class="filter-section">
                <div class="filter-section-title">General</div>
                <label><input type="checkbox" id="diffFilterDirectives" onchange="applyDiffFilters()"> Hide directives</label>
                <label><input type="checkbox" id="diffFilterComments" onchange="applyDiffFilters()"> Hide comments</label>
                <label><input type="checkbox" id="diffFilterEmpty" onchange="applyDiffFilters()"> Hide empty lines</label>
            </div>
            <div class="filter-section" id="diffInstructionFilters">
                <div class="filter-section-title">Instructions</div>
                <div class="filter-search-container">
                    <input type="text" id="diffInstrSearchBox" placeholder="Search instructions..." onkeyup="filterDiffInstructionList()" />
                </div>
                <div class="filter-actions">
                    <button class="filter-action-btn" onclick="selectAllDiffInstructions()">Select All</button>
                    <button class="filter-action-btn" onclick="selectNoneDiffInstructions()">Select None</button>
                </div>
                <div id="diffInstructionList">
                    <!-- Dynamic instruction filters will be inserted here -->
                </div>
            </div>
        </div>
    </div>
    <div class="search-bar" id="diffSearchBar" style="display: none;">
        <input type="text" id="diffSearchInput" placeholder="Find in diff..." />
        <button class="search-nav-btn" onclick="findDiffPrevious()" title="Previous match (Shift+F3)">▲</button>
        <button class="search-nav-btn" onclick="findDiffNext()" title="Next match (F3)">▼</button>
        <span id="diffSearchResults" class="search-results"></span>
        <button class="search-close-btn" onclick="toggleDiffSearch()">✕</button>
    </div>

    <div class="diff-container">
        <div class="diff-side" id="left-side">
            <div class="diff-header">${leftLabel} (${this.escapeHtml(displayText1)})</div>
            ${leftHtml}
        </div>
        <div class="diff-side" id="right-side">
            <div class="diff-header">${rightLabel} (${this.escapeHtml(displayText2)})</div>
            ${rightHtml}
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        // Synchronized scrolling
        const leftSide = document.getElementById('left-side');
        const rightSide = document.getElementById('right-side');

        let isScrolling = false;

        leftSide.addEventListener('scroll', () => {
            if (!isScrolling) {
                isScrolling = true;
                rightSide.scrollTop = leftSide.scrollTop;
                setTimeout(() => isScrolling = false, 50);
            }
        });

        rightSide.addEventListener('scroll', () => {
            if (!isScrolling) {
                isScrolling = true;
                leftSide.scrollTop = rightSide.scrollTop;
                setTimeout(() => isScrolling = false, 50);
            }
        });

        // Click to jump to source
        document.querySelectorAll('.asm-line').forEach(line => {
            line.addEventListener('click', () => {
                // Clear previous highlights
                document.querySelectorAll('.asm-line.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });

                // Highlight clicked line
                line.classList.add('highlighted');

                // Send message to extension to highlight source
                const lineNum = parseInt(line.dataset.line);
                const side = line.dataset.side;
                vscode.postMessage({
                    command: 'highlightSourceFromDiff',
                    line: lineNum,
                    side: side
                });
            });
        });

        function closeDiff() {
            vscode.postMessage({ command: 'closeDiff' });
        }

        // Filter menu functions
        function toggleDiffFilterMenu(event) {
            const menu = document.getElementById('diffFilterMenu');
            if (menu.style.display === 'none') {
                const btn = event?.target?.closest('.icon-btn');
                if (btn) {
                    const rect = btn.getBoundingClientRect();
                    menu.style.top = (rect.bottom + 4) + 'px';
                    menu.style.left = 'auto';
                    menu.style.right = (window.innerWidth - rect.right) + 'px';
                }
                menu.style.display = 'block';
            } else {
                menu.style.display = 'none';
            }
        }

        // Build dynamic instruction filters
        function buildDiffInstructionFilters() {
            const instructionCounts = {};

            document.querySelectorAll('.asm-line').forEach(line => {
                line.classList.forEach(cls => {
                    if (cls.startsWith('instr-')) {
                        const instr = cls.substring(6);
                        instructionCounts[instr] = (instructionCounts[instr] || 0) + 1;
                    }
                });
            });

            const sortedInstructions = Object.keys(instructionCounts).sort();
            const instructionListDiv = document.getElementById('diffInstructionList');
            if (instructionListDiv && sortedInstructions.length > 0) {
                let html = '';
                sortedInstructions.forEach(instr => {
                    const count = instructionCounts[instr];
                    const id = 'diff-filter-instr-' + instr.replace(/[^a-z0-9]/g, '_');
                    html += '<label data-instr-name="' + instr.toLowerCase() + '"><input type="checkbox" id="' + id + '" data-instr="' + instr + '" onchange="applyDiffFilters()"> ' + instr + ' (' + count + ')</label>';
                });
                instructionListDiv.innerHTML = html;
            }

            const titleDiv = document.querySelector('#diffInstructionFilters .filter-section-title');
            if (titleDiv) {
                titleDiv.textContent = 'Instructions (' + sortedInstructions.length + ')';
            }
        }

        function filterDiffInstructionList() {
            const searchText = document.getElementById('diffInstrSearchBox').value.toLowerCase();
            const labels = document.querySelectorAll('#diffInstructionList label');

            labels.forEach(label => {
                const instrName = label.getAttribute('data-instr-name');
                if (instrName.includes(searchText)) {
                    label.classList.remove('filter-hidden');
                } else {
                    label.classList.add('filter-hidden');
                }
            });
        }

        function selectAllDiffInstructions() {
            const checkboxes = document.querySelectorAll('#diffInstructionList label:not(.filter-hidden) input[type="checkbox"]');
            checkboxes.forEach(cb => cb.checked = true);
            applyDiffFilters();
        }

        function selectNoneDiffInstructions() {
            const checkboxes = document.querySelectorAll('#diffInstructionList input[type="checkbox"]');
            checkboxes.forEach(cb => cb.checked = false);
            applyDiffFilters();
        }

        function applyDiffFilters() {
            const hideDirectives = document.getElementById('diffFilterDirectives').checked;
            const hideComments = document.getElementById('diffFilterComments').checked;
            const hideEmpty = document.getElementById('diffFilterEmpty').checked;

            const instrCheckboxes = document.querySelectorAll('#diffInstructionList input[type="checkbox"]');
            const hiddenInstructions = new Set();
            instrCheckboxes.forEach(cb => {
                if (cb.checked) {
                    hiddenInstructions.add('instr-' + cb.getAttribute('data-instr'));
                }
            });

            document.querySelectorAll('.asm-line').forEach(line => {
                line.classList.remove('hidden');

                let isHidden = false;
                if (hideDirectives && line.classList.contains('filter-directive')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }
                if (hideComments && line.classList.contains('filter-comment')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }
                if (hideEmpty && line.classList.contains('filter-empty')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }

                if (!isHidden) {
                    line.classList.forEach(cls => {
                        if (hiddenInstructions.has(cls)) {
                            line.classList.add('hidden');
                            isHidden = true;
                        }
                    });
                }
            });

            // Save filter state for persistence
            const hiddenInstrArray = [];
            instrCheckboxes.forEach(cb => {
                if (cb.checked) {
                    hiddenInstrArray.push(cb.getAttribute('data-instr'));
                }
            });

            vscode.postMessage({
                command: 'saveFilterState',
                filterState: {
                    hideDirectives: hideDirectives,
                    hideComments: hideComments,
                    hideEmpty: hideEmpty,
                    hiddenInstructions: hiddenInstrArray
                }
            });
        }

        // Close filter menu when clicking outside
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('diffFilterMenu');
            const filterBtn = document.getElementById('diffFilterBtn');

            if (menu && filterBtn && !menu.contains(e.target) && !filterBtn.contains(e.target)) {
                menu.style.display = 'none';
            }
        });

        // Initialize instruction filters after page load
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, building instruction filters...');
            buildDiffInstructionFilters();
        });
        setTimeout(() => {
            console.log('Timeout building instruction filters...');
            buildDiffInstructionFilters();
        }, 100);

        // Inherit filter state from main view
        const inheritedFilters = ${inheritedFilterState};
        console.log('Inherited filter state on diff view load:', inheritedFilters);

        // Function to apply inherited filters
        function applyInheritedFilters() {
            if (!inheritedFilters) {
                console.log('No inherited filters to apply');
                return;
            }

            console.log('Applying inherited filters:', inheritedFilters);

            // Apply inherited general filters
            if (inheritedFilters.hideDirectives) {
                document.getElementById('diffFilterDirectives').checked = true;
            }
            if (inheritedFilters.hideComments) {
                document.getElementById('diffFilterComments').checked = true;
            }
            if (inheritedFilters.hideEmpty) {
                document.getElementById('diffFilterEmpty').checked = true;
            }

            // Apply inherited instruction filters (wait for instruction list to be built)
            if (inheritedFilters.hiddenInstructions && inheritedFilters.hiddenInstructions.length > 0) {
                let retries = 0;
                const applyInstrFilters = () => {
                    let appliedCount = 0;
                    inheritedFilters.hiddenInstructions.forEach(instr => {
                        const id = 'diff-filter-instr-' + instr.replace(/[^a-z0-9]/g, '_');
                        const checkbox = document.getElementById(id);
                        if (checkbox) {
                            checkbox.checked = true;
                            appliedCount++;
                        }
                    });

                    console.log('Applied ' + appliedCount + ' instruction filters out of ' + inheritedFilters.hiddenInstructions.length);

                    // If not all filters were applied and we haven't retried too many times, try again
                    if (appliedCount < inheritedFilters.hiddenInstructions.length && retries < 10) {
                        retries++;
                        setTimeout(applyInstrFilters, 100);
                    } else {
                        // Apply filters once all checkboxes are set
                        applyDiffFilters();
                    }
                };

                setTimeout(applyInstrFilters, 150);
            } else {
                // Apply general filters immediately if no instruction filters
                applyDiffFilters();
            }
        }

        // Apply inherited filters after instruction list is built
        // Wait longer to ensure instruction list is fully populated
        setTimeout(() => {
            console.log('Starting to apply inherited filters...');
            applyInheritedFilters();
        }, 300);

        // Search functionality
        let diffSearchMatches = [];
        let diffCurrentMatchIndex = -1;

        function toggleDiffSearch() {
            const searchBar = document.getElementById('diffSearchBar');
            const searchInput = document.getElementById('diffSearchInput');

            if (searchBar.style.display === 'none') {
                searchBar.style.display = 'flex';
                searchInput.focus();
                searchInput.select();
            } else {
                searchBar.style.display = 'none';
                clearDiffSearch();
            }
        }

        function clearDiffSearch() {
            document.querySelectorAll('.search-match, .search-match-current').forEach(el => {
                el.classList.remove('search-match', 'search-match-current');
            });
            diffSearchMatches = [];
            diffCurrentMatchIndex = -1;
            document.getElementById('diffSearchResults').textContent = '';
        }

        function performDiffSearch() {
            const searchText = document.getElementById('diffSearchInput').value;
            clearDiffSearch();

            if (!searchText) return;

            const lines = document.querySelectorAll('.asm-line .asm-content');
            lines.forEach((line, index) => {
                const text = line.textContent;
                if (text.toLowerCase().includes(searchText.toLowerCase())) {
                    line.parentElement.classList.add('search-match');
                    diffSearchMatches.push(line.parentElement);
                }
            });

            if (diffSearchMatches.length > 0) {
                diffCurrentMatchIndex = 0;
                highlightDiffCurrentMatch();
                updateDiffSearchResults();
            } else {
                document.getElementById('diffSearchResults').textContent = 'No results';
            }
        }

        function highlightDiffCurrentMatch() {
            diffSearchMatches.forEach((match, index) => {
                if (index === diffCurrentMatchIndex) {
                    match.classList.add('search-match-current');
                    match.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    match.classList.remove('search-match-current');
                }
            });
        }

        function updateDiffSearchResults() {
            if (diffSearchMatches.length > 0) {
                document.getElementById('diffSearchResults').textContent =
                    (diffCurrentMatchIndex + 1) + ' of ' + diffSearchMatches.length;
            }
        }

        function findDiffNext() {
            if (diffSearchMatches.length === 0) return;
            diffCurrentMatchIndex = (diffCurrentMatchIndex + 1) % diffSearchMatches.length;
            highlightDiffCurrentMatch();
            updateDiffSearchResults();
        }

        function findDiffPrevious() {
            if (diffSearchMatches.length === 0) return;
            diffCurrentMatchIndex = (diffCurrentMatchIndex - 1 + diffSearchMatches.length) % diffSearchMatches.length;
            highlightDiffCurrentMatch();
            updateDiffSearchResults();
        }

        // Search input handler
        document.addEventListener('DOMContentLoaded', () => {
            const searchInput = document.getElementById('diffSearchInput');
            if (searchInput) {
                searchInput.addEventListener('input', performDiffSearch);
                searchInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        if (e.shiftKey) {
                            findDiffPrevious();
                        } else {
                            findDiffNext();
                        }
                    } else if (e.key === 'Escape') {
                        toggleDiffSearch();
                    }
                });
            }
        });

        // Keyboard shortcuts for search
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                toggleDiffSearch();
            } else if (e.key === 'F3') {
                e.preventDefault();
                if (e.shiftKey) {
                    findDiffPrevious();
                } else {
                    findDiffNext();
                }
            }
        });
    </script>
</body>
</html>`;
    }

    private async highlightSourceLine(asmLine: number, kernelIndex?: number) {
        // Find which source line(s) and file correspond to this assembly line
        let sourceLine = -1;
        let sourceFile: string | undefined;
        let targetAsm: KernelAssembly | undefined;
        let localLineIndex = asmLine;
        let globalLineNum = asmLine;

        // Use _selectedKernelIndex to determine which kernel we're in
        const selectedIdx = this._selectedKernelIndex ?? 0;
        if (selectedIdx >= 0 && selectedIdx < this._allAssemblies.length) {
            targetAsm = this._allAssemblies[selectedIdx];
            localLineIndex = asmLine;

            // Calculate global line number by adding offsets from previous kernels
            let globalLineOffset = 0;
            for (let i = 0; i < selectedIdx; i++) {
                globalLineOffset += this._allAssemblies[i].assembly.split('\n').length;
            }
            globalLineNum = globalLineOffset + asmLine;

            // Try to get source file from mapping using global line number
            sourceFile = this._lineMapping.asmLineToFile.get(globalLineNum);

            logger.log(`[Trident] Assembly line ${asmLine} (local) -> ${globalLineNum} (global) in kernel ${targetAsm.kernelName}`);
        }

        if (targetAsm && localLineIndex >= 0) {
            const asmLines = targetAsm.assembly.split('\n');

            if (targetAsm.irType === 'ttgir' || targetAsm.irType === 'ttir') {
                // TTGIR and TTIR: Walk backwards to find most recent #loc = loc("filepath":line:col)
                // Also handle loc(#locN) references
                const locPattern1 = /#loc\s*=\s*loc\("([^"]+)":(\d+):\d+\)/;
                const locPattern2 = /loc\("([^"]+)":(\d+):\d+\)/;
                const locDefPattern = /#loc(\d+)\s*=\s*loc\("([^"]+)":(\d+):\d+\)/;
                const locRefPattern = /loc\(#loc(\d+)\)/;

                // First, collect all location definitions
                const locDefMap = new Map<number, { file: string; line: number }>();
                for (let i = 0; i < asmLines.length; i++) {
                    const defMatch = asmLines[i].match(locDefPattern);
                    if (defMatch) {
                        const locId = parseInt(defMatch[1]);
                        const filePath = defMatch[2];
                        const lineNum = parseInt(defMatch[3]) - 1; // Convert to 0-indexed
                        locDefMap.set(locId, { file: filePath, line: lineNum });
                    }
                }

                logger.log(`[Trident] ${targetAsm.irType.toUpperCase()}: Looking for source line at assembly line ${localLineIndex} in ${targetAsm.kernelName}`);

                for (let i = localLineIndex; i >= 0; i--) {
                    const line = asmLines[i];
                    let match = line.match(locPattern1);
                    if (!match) {
                        match = line.match(locPattern2);
                    }
                    if (!match) {
                        // Try location reference
                        const refMatch = line.match(locRefPattern);
                        if (refMatch) {
                            const locId = parseInt(refMatch[1]);
                            const locDef = locDefMap.get(locId);
                            if (locDef) {
                                if (!sourceFile || path.normalize(locDef.file) === path.normalize(sourceFile)) {
                                    sourceFile = locDef.file;
                                    sourceLine = locDef.line;
                                    logger.log(`[Trident] ${targetAsm.irType.toUpperCase()}: Found source line ${sourceLine + 1} in ${sourceFile} via loc(#loc${locId})`);
                                    break;
                                }
                            }
                        }
                    } else {
                        const filePath = match[1];
                        if (!sourceFile || path.normalize(filePath) === path.normalize(sourceFile)) {
                            sourceFile = filePath;
                            sourceLine = parseInt(match[2]) - 1; // Convert to 0-indexed
                            logger.log(`[Trident] ${targetAsm.irType.toUpperCase()}: Found source line ${sourceLine + 1} in ${filePath}`);
                            break;
                        }
                    }
                }
            } else if (targetAsm.irType === 'llvm') {
                // LLVM IR: Walk backwards to find most recent !dbg !N reference
                // Then look up the !DILocation definition (already collected in buildLineMappingLLVM)
                // We need to rebuild the maps for this specific assembly
                const difilePattern = /!(\d+)\s*=\s*!DIFile\([^)]*filename:\s*"([^"]+)"[^)]*directory:\s*"([^"]+)"[^)]*\)/;
                const disubprogramPattern = /!(\d+)\s*=\s*(?:distinct\s+)?!DISubprogram\([^)]*\)/;
                const disubprogramFilePattern = /file:\s*!(\d+)/;
                const disubprogramLinePattern = /line:\s*(\d+)/;
                const dilocationPattern = /!(\d+)\s*=\s*!DILocation\([^)]*line:\s*(\d+)[^)]*scope:\s*!(\d+)[^)]*\)/;

                // First pass: collect all !DIFile and !DISubprogram metadata
                const fileMap = new Map<number, string>();
                const subprogramMap = new Map<number, { file: string; line: number }>();
                const locationMap = new Map<number, { line: number; scopeId: number }>();

                for (let i = 0; i < asmLines.length; i++) {
                    const fileMatch = asmLines[i].match(difilePattern);
                    if (fileMatch) {
                        const fileId = parseInt(fileMatch[1]);
                        const filename = fileMatch[2];
                        const directory = fileMatch[3];
                        fileMap.set(fileId, path.join(directory, filename));
                    }

                    const subprogramMatch = asmLines[i].match(disubprogramPattern);
                    if (subprogramMatch) {
                        const subprogramId = parseInt(subprogramMatch[1]);
                        const fileMatch = asmLines[i].match(disubprogramFilePattern);
                        const lineMatch = asmLines[i].match(disubprogramLinePattern);
                        if (fileMatch && lineMatch) {
                            const fileId = parseInt(fileMatch[1]);
                            const lineNum = parseInt(lineMatch[1]) - 1;
                            const filePath = fileMap.get(fileId);
                            if (filePath) {
                                subprogramMap.set(subprogramId, { file: filePath, line: lineNum });
                            }
                        }
                    }

                    const locMatch = asmLines[i].match(dilocationPattern);
                    if (locMatch) {
                        const locationId = parseInt(locMatch[1]);
                        const lineNum = parseInt(locMatch[2]) - 1;
                        const scopeId = parseInt(locMatch[3]);
                        locationMap.set(locationId, { line: lineNum, scopeId: scopeId });
                    }
                }

                // Second pass: walk backwards to find !dbg !N and resolve !DILocation
                for (let i = localLineIndex; i >= 0; i--) {
                    const line = asmLines[i];
                    const dbgMatch = line.match(/!dbg\s*!(\d+)/);
                    if (dbgMatch) {
                        const locationId = parseInt(dbgMatch[1]);
                        const locationDef = locationMap.get(locationId);
                        if (locationDef) {
                            const subprogram = subprogramMap.get(locationDef.scopeId);
                            if (subprogram) {
                                if (!sourceFile || path.normalize(subprogram.file) === path.normalize(sourceFile)) {
                                    sourceFile = subprogram.file;
                                    sourceLine = locationDef.line;
                                    logger.log(`[Trident] LLVM: Found source line ${sourceLine + 1} in ${sourceFile}`);
                                    break;
                                }
                            }
                        }
                    }
                }
            } else if (targetAsm.irType === 'gcn') {
                // GCN: Walk backwards to find most recent .file and .loc directives
                let currentFileId = -1;
                const fileMap = new Map<number, string>();

                for (let i = 0; i <= localLineIndex; i++) {
                    const line = asmLines[i];

                    // Parse .file directive
                    const format2Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"/);
                    if (format2Match) {
                        const fileId = parseInt(format2Match[1]);
                        const directory = format2Match[2].replace(/^;/, '');
                        const filename = format2Match[3];
                        fileMap.set(fileId, path.join(directory, filename));
                        continue;
                    }

                    const format1Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"/);
                    if (format1Match) {
                        const fileId = parseInt(format1Match[1]);
                        fileMap.set(fileId, format1Match[2]);
                        continue;
                    }

                    // Parse .loc directive
                    const locMatch = line.match(/\.loc\s+(\d+)\s+(\d+)\s+\d+/);
                    if (locMatch) {
                        currentFileId = parseInt(locMatch[1]);
                        sourceLine = parseInt(locMatch[2]) - 1; // Convert to 0-indexed
                    }
                }

                if (currentFileId >= 0 && fileMap.has(currentFileId)) {
                    sourceFile = fileMap.get(currentFileId);
                }
            }
        }

        if (sourceLine >= 0 && sourceFile) {
            try {
                // Try to open the file
                const uri = vscode.Uri.file(sourceFile);
                const document = await vscode.workspace.openTextDocument(uri);
                const editor = await vscode.window.showTextDocument(document, {
                    viewColumn: vscode.ViewColumn.One,
                    preserveFocus: true  // Keep focus on assembly viewer
                });

                // Highlight and reveal the source line
                if (sourceLine < document.lineCount) {
                    // Set flag to prevent circular highlighting (assembly->source->assembly)
                    AssemblyViewerPanel._isProgrammaticSelection = true;

                    const range = document.lineAt(sourceLine).range;
                    editor.selection = new vscode.Selection(range.start, range.start);
                    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);

                    // Apply highlight decoration
                    this._highlightManager.highlightSource(editor, [sourceLine]);

                    // Clear flag after a short delay
                    setTimeout(() => {
                        AssemblyViewerPanel._isProgrammaticSelection = false;
                    }, 200);
                }
            } catch (error) {
                vscode.window.showWarningMessage(`Could not open file: ${sourceFile}`);
                console.error('Error opening source file:', error);
            }
        }
    }

    private async getWebviewContent(): Promise<string> {
        logger.log(`Building webview with ${this._assemblies.length} kernels to display:`);
        this._assemblies.forEach((asm, i) => {
            logger.log(`  [${i}] ${asm.kernelName}`);
        });

        // Find the latest kernel for each kernel name from ALL assemblies (not just filtered)
        const latestByName = new Map<string, Date>();
        this._allAssemblies.forEach(asm => {
            const existing = latestByName.get(asm.kernelName);
            if (!existing || asm.timestamp > existing) {
                latestByName.set(asm.kernelName, asm.timestamp);
            }
        });

        // Always show only the selected kernel in the assembly view
        // But keep all kernels in the dropdown (_assemblies)
        let assembliesToDisplay: KernelAssembly[] = [];
        const selectedIdx = this._selectedKernelIndex ?? -1;
        const allAsms = this._allAssemblies ?? [];

        if (selectedIdx >= 0 && selectedIdx < allAsms.length) {
            // A kernel is selected - show only that one
            assembliesToDisplay = [allAsms[selectedIdx]];
        } else if (this._assemblies.length > 0) {
            // No kernel selected - default to first filtered kernel
            const firstIndex = allAsms.indexOf(this._assemblies[0]);
            if (firstIndex >= 0) {
                this._selectedKernelIndex = firstIndex;
                assembliesToDisplay = [this._assemblies[0]];
            }
        }

        const assembliesHtml = assembliesToDisplay.map((asm, index) => {
            const lines = asm.assembly.split('\n');
            const isLatest = latestByName.get(asm.kernelName)?.getTime() === asm.timestamp.getTime();

            const numberedLines = lines.map((line, i) => {
                const lineNum = (i + 1).toString().padStart(4, ' ');
                const trimmedLine = line.trim();
                // Directive detection: starts with . OR starts with - (YAML list) OR starts with amdhsa
                const isDirective = trimmedLine.startsWith('.') ||
                    trimmedLine.startsWith('-') ||
                    trimmedLine.startsWith('amdhsa') ||
                    trimmedLine === '---' ||
                    trimmedLine === '...';
                const isComment = trimmedLine.startsWith('#') || trimmedLine.startsWith(';') || trimmedLine.startsWith('//');
                const isEmpty = trimmedLine.length === 0;

                const filterClasses = [];
                if (isDirective) {
                    filterClasses.push('filter-directive');
                }
                if (isComment) {
                    filterClasses.push('filter-comment');
                }
                if (isEmpty) {
                    filterClasses.push('filter-empty');
                }

                // Extract instruction name and add as class
                if (!isDirective && !isComment && !isEmpty) {
                    const instrMatch = trimmedLine.match(/^\s*(\w+)/);
                    if (instrMatch) {
                        const instr = instrMatch[1].toLowerCase();

                        // Filter out labels (ending with :) and amdhsa/metadata keywords
                        const isLabel = trimmedLine.trim().endsWith(':');
                        const isAmdHsa = instr.startsWith('amdhsa') || instr === 'end_amd_kernel_code_t';
                        const isMetadata = instr === 'amd_kernel_code_t' || instr === 'kernel_code_entry_byte_offset' ||
                            instr === 'kernel_code_prefetch_byte_size' || instr === 'max_scratch_backing_memory_byte_size';

                        if (!isLabel && !isAmdHsa && !isMetadata) {
                            filterClasses.push('instr-' + instr);
                        }
                    }
                }

                return `<div class="asm-line clickable ${filterClasses.join(' ')}" data-line="${i}" data-kernel-index="${index}"><span class="line-number">${lineNum}</span><span class="asm-content">${this.highlightGCNSyntax(line)}</span></div>`;
            }).join('');

            const displayText = this.getKernelDisplayText(asm);
            const fullDate = asm.timestamp.toLocaleString();
            const customLabel = this.getKernelLabel(asm.cachePath);
            const timestampStr = this.formatTimestamp(asm.timestamp);
            const outdatedWarning = isLatest ? '' : `
                <div class="outdated-warning">
                    ⚠️ This is an older version. Source code may have changed. Latest version: ${this.formatTimestamp(latestByName.get(asm.kernelName)!)}
                </div>
            `;

            return `
                <div class="kernel-section ${isLatest ? '' : 'outdated-kernel'}" id="kernel-${index}" data-kernel-name="${this.escapeHtml(asm.kernelName)}" data-is-latest="${isLatest}">
                    <div class="kernel-header">
                        <button class="icon-btn edit-label-btn" onclick="editLabel('${this.escapeHtml(asm.cachePath)}', '${this.escapeHtml(customLabel || '')}')" title="Edit label">
                            <i class="codicon codicon-edit"></i>
                        </button>
                        <span class="kernel-title">
                            <span class="ir-type-badge ir-type-${asm.irType}">${asm.irType.toUpperCase()}</span>
                            ${this.escapeHtml(asm.kernelName)}
                            <span class="kernel-date">(${this.escapeHtml(displayText)})</span>
                        </span>
                        <span class="cache-info" title="${fullDate} - Click to open file" onclick="openCacheFile('${this.escapeHtml(asm.cachePath)}')">${asm.cachePath}</span>
                    </div>
                    ${outdatedWarning}
                    <div class="assembly-code">
                        ${numberedLines}
                    </div>
                </div>
            `;
        }).join('');

        // Get unique IR types from all assemblies
        const availableIRTypes = Array.from(new Set(this._allAssemblies.map(asm => asm.irType))).sort();
        console.log(`[Trident UI] Total assemblies: ${this._allAssemblies.length}`);
        console.log(`[Trident UI] Available IR types:`, availableIRTypes);
        console.log(`[Trident UI] Assemblies by type:`, this._allAssemblies.reduce((acc, asm) => {
            acc[asm.irType] = (acc[asm.irType] || 0) + 1;
            return acc;
        }, {} as Record<string, number>));
        const irTypeOptions = availableIRTypes.map(irType =>
            `<option value="${irType}"${this._selectedIRType === irType ? ' selected' : ''}>${irType.toUpperCase()}</option>`
        ).join('');

        // Build kernel selector options with custom labels or timestamps (from filtered assemblies)
        // Use the actual index in _allAssemblies as the value, not the filtered index
        const kernelOptions = this._assemblies.map((asm) => {
            const actualIndex = this._allAssemblies.indexOf(asm);
            const displayText = this.getKernelDisplayText(asm);
            const selected = this._selectedKernelIndex === actualIndex ? ' selected' : '';
            return `<option value="${actualIndex}"${selected}>${this.escapeHtml(asm.kernelName)} [${asm.irType.toUpperCase()}] (${this.escapeHtml(displayText)})</option>`;
        }).join('');

        // Build source file selector options (unique files)
        const allSourceFiles = new Set<string>();

        // Add files from current assemblies
        this._assemblies.forEach(asm => {
            asm.sourceFiles.forEach(file => allSourceFiles.add(file));
        });

        // Add ALL open Triton files from our cache
        this._allOpenTritonFiles.forEach(file => allSourceFiles.add(file));

        // Also get all source files from the cache (not just open files)
        try {
            // Extract kernel names from all open Python files
            const kernelNames = new Set<string>();
            for (const doc of vscode.workspace.textDocuments) {
                if (doc.languageId === 'python' && !doc.isClosed) {
                    const content = doc.getText();
                    if (content.includes('@triton.jit')) {
                        const names = this._compiler.extractKernelNames(content);
                        names.forEach(name => kernelNames.add(name));
                    }
                }
            }
            const workspaceFolders = vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath) || [];
            const allAssemblies = kernelNames.size > 0 ? await this._compiler.findCachedAssemblies(kernelNames, workspaceFolders) : [];

            // Extract all unique source files from all assemblies
            for (const asm of allAssemblies) {
                for (const sourceFile of asm.sourceFiles) {
                    allSourceFiles.add(sourceFile);
                }
            }
        } catch (error) {
            logger.error('Error scanning cache for source files in getWebviewContent', error);
        }

        // Only show open Python files in dropdown
        const activeEditor = vscode.window.activeTextEditor;
        const activeFile = activeEditor?.document.languageId === 'python' ? activeEditor.document.uri.fsPath : undefined;
        const openPythonFiles = vscode.workspace.textDocuments
            .filter(doc => doc.languageId === 'python' && !doc.isClosed)
            .map(doc => doc.uri.fsPath);

        logger.log(`Building dropdown with ${openPythonFiles.length} open files: ${openPythonFiles.map(f => path.basename(f)).join(', ')}`);

        // Categorize: current file first, then other open files
        const currentFile = activeFile && openPythonFiles.includes(activeFile) ? activeFile : undefined;
        const otherOpenFiles = openPythonFiles.filter(f => f !== currentFile);

        // Build options with styling
        let sourceFileOptions = '';

        // Currently visible file - normal styling
        if (currentFile) {
            const selected = this._selectedSourceFile === currentFile ? ' selected' : '';
            sourceFileOptions += `<option value="${this.escapeHtml(currentFile)}"${selected}>${this.escapeHtml(path.basename(currentFile))}</option>`;
        }

        // Other open files - slightly dimmed
        for (const file of otherOpenFiles) {
            const selected = this._selectedSourceFile === file ? ' selected' : '';
            sourceFileOptions += `<option value="${this.escapeHtml(file)}"${selected} style="opacity: 0.7;">${this.escapeHtml(path.basename(file))}</option>`;
        }

        logger.log(`Generated dropdown: current=${currentFile ? path.basename(currentFile) : 'none'}, other open=${otherOpenFiles.length}`);

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${this._panel.webview.cspSource} 'unsafe-inline' https:; script-src 'unsafe-inline'; font-src ${this._panel.webview.cspSource} https:;">
    <link href="https://unpkg.com/@vscode/codicons@latest/dist/codicon.css" rel="stylesheet" />
    <title>Trident Assembly</title>
    <style>
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            padding: 0;
            margin: 0;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }

        .toolbar {
            position: sticky;
            top: 0;
            background-color: var(--vscode-editorWidget-background);
            border-bottom: 1px solid var(--vscode-widget-border);
            padding: 6px 12px;
            display: flex;
            gap: 6px;
            align-items: center;
            z-index: 100;
        }

        .search-bar {
            position: sticky;
            top: 44px;
            background-color: var(--vscode-editorWidget-background);
            border-bottom: 1px solid var(--vscode-widget-border);
            padding: 6px 12px;
            display: flex;
            gap: 6px;
            align-items: center;
            z-index: 99;
        }

        .search-bar input {
            flex: 1;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 13px;
            outline: none;
        }

        .search-bar input:focus {
            border-color: var(--vscode-focusBorder);
        }

        .search-nav-btn, .search-close-btn {
            background-color: transparent;
            color: var(--vscode-foreground);
            border: none;
            padding: 4px 8px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 12px;
        }

        .search-nav-btn:hover, .search-close-btn:hover {
            background-color: var(--vscode-toolbar-hoverBackground);
        }

        .search-results {
            color: var(--vscode-descriptionForeground);
            font-size: 12px;
            white-space: nowrap;
        }

        .search-match {
            background-color: rgba(255, 230, 100, 0.35) !important;
            outline: 2px solid rgba(255, 200, 0, 0.6);
            outline-offset: -2px;
        }

        .search-match-current {
            background-color: rgba(255, 140, 0, 0.5) !important;
            outline: 2px solid rgba(255, 100, 0, 0.9);
            outline-offset: -2px;
        }

        /* Search matches should be visible even on diff-changed lines */
        .asm-line.diff-changed.search-match {
            background: linear-gradient(rgba(255, 230, 100, 0.4), rgba(255, 230, 100, 0.4)), var(--vscode-diffEditor-insertedTextBackground, rgba(155, 185, 85, 0.2)) !important;
        }

        .asm-line.diff-changed.search-match-current {
            background: linear-gradient(rgba(255, 140, 0, 0.6), rgba(255, 140, 0, 0.6)), var(--vscode-diffEditor-insertedTextBackground, rgba(155, 185, 85, 0.2)) !important;
        }

        .diff-side:first-child .asm-line.diff-changed.search-match {
            background: linear-gradient(rgba(255, 230, 100, 0.4), rgba(255, 230, 100, 0.4)), var(--vscode-diffEditor-removedTextBackground, rgba(255, 0, 0, 0.2)) !important;
        }

        .diff-side:first-child .asm-line.diff-changed.search-match-current {
            background: linear-gradient(rgba(255, 140, 0, 0.6), rgba(255, 140, 0, 0.6)), var(--vscode-diffEditor-removedTextBackground, rgba(255, 0, 0, 0.2)) !important;
        }

        .btn {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 6px 12px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 13px;
        }

        .btn:hover:not(:disabled) {
            background-color: var(--vscode-button-hoverBackground);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .icon-btn {
            background-color: transparent;
            color: var(--vscode-foreground);
            border: none;
            padding: 4px 6px;
            cursor: pointer;
            border-radius: 2px;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 32px;
            height: 32px;
        }

        .icon-btn:hover:not(:disabled) {
            background-color: var(--vscode-toolbar-hoverBackground);
        }

        .icon-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .icon-btn .codicon {
            font-size: 16px;
        }

        .toolbar-separator {
            width: 1px;
            height: 24px;
            background-color: var(--vscode-widget-border);
            margin: 0 4px;
        }

        .kernel-selector {
            background-color: var(--vscode-dropdown-background);
            color: var(--vscode-dropdown-foreground);
            border: 1px solid var(--vscode-dropdown-border);
            padding: 5px 8px;
            border-radius: 2px;
            font-size: 13px;
            cursor: pointer;
            min-width: 180px;
            height: 32px;
        }

        .kernel-selector:hover {
            background-color: var(--vscode-dropdown-listBackground);
        }

        .filter-menu {
            position: fixed;
            background-color: var(--vscode-dropdown-background);
            border: 1px solid var(--vscode-dropdown-border);
            border-radius: 2px;
            padding: 8px;
            z-index: 1000;
            min-width: 300px;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .filter-section {
            margin-bottom: 12px;
        }

        .filter-section:last-child {
            margin-bottom: 0;
        }

        .filter-section-title {
            font-weight: bold;
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 6px;
            padding: 0 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .filter-menu label {
            display: block;
            padding: 4px 0;
            cursor: pointer;
            color: var(--vscode-dropdown-foreground);
            user-select: none;
        }

        .filter-menu label:hover {
            background-color: var(--vscode-list-hoverBackground);
        }

        .filter-menu input[type="checkbox"] {
            margin-right: 8px;
            cursor: pointer;
        }

        .filter-search-container {
            padding: 4px 8px 8px 8px;
        }

        .filter-search-container input {
            width: 100%;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 12px;
            outline: none;
        }

        .filter-search-container input:focus {
            border-color: var(--vscode-focusBorder);
        }

        .filter-actions {
            display: flex;
            gap: 4px;
            padding: 0 8px 8px 8px;
        }

        .filter-action-btn {
            flex: 1;
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            border: none;
            padding: 4px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 2px;
        }

        .filter-action-btn:hover {
            background-color: var(--vscode-button-secondaryHoverBackground);
        }

        #instructionList {
            max-height: 300px;
            overflow-y: auto;
        }

        #instructionList label {
            display: block;
        }

        #instructionList label.filter-hidden {
            display: none;
        }

        .content {
            padding: 16px;
        }

        .kernel-section {
            margin-bottom: 30px;
            border: 1px solid var(--vscode-widget-border);
            border-radius: 4px;
            overflow: hidden;
        }

        .kernel-header {
            display: flex;
            align-items: center;
            gap: 8px;
            background-color: var(--vscode-editorWidget-background);
            padding: 12px 16px;
            border-bottom: 1px solid var(--vscode-widget-border);
        }

        .kernel-title {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            font-weight: bold;
            color: var(--vscode-editor-foreground);
            white-space: nowrap;
        }

        .kernel-date {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            font-weight: normal;
            color: var(--vscode-descriptionForeground);
            opacity: 0.7;
        }

        .ir-type-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            margin-right: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .ir-type-gcn {
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
        }


        .ir-type-llvm {
            background-color: rgba(33, 150, 243, 0.2);
            color: #2196f3;
        }

        .ir-type-ttgir {
            background-color: rgba(156, 39, 176, 0.2);
            color: #9c27b0;
        }

        .ir-type-ttir {
            background-color: rgba(76, 175, 80, 0.2);
            color: #4caf50;
        }

        .ir-type-unknown {
            background-color: rgba(158, 158, 158, 0.2);
            color: #9e9e9e;
        }

        .cache-info {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            color: var(--vscode-textLink-foreground);
            opacity: 0.7;
            margin-left: auto;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            text-decoration: underline;
            text-decoration-style: dotted;
        }

        .cache-info:hover {
            opacity: 1;
            text-decoration-style: solid;
        }

        .assembly-code {
            background-color: var(--vscode-editor-background);
            padding: 12px;
            overflow-x: auto;
        }

        .asm-line {
            padding: 2px 0;
            font-size: 13px;
            line-height: 1.6;
            display: block;
            white-space: pre;
        }

        .line-number {
            color: var(--vscode-editorLineNumber-foreground);
            padding: 0 12px 0 4px;
            user-select: none;
            font-family: monospace;
            display: inline-block;
            width: 60px;
            text-align: right;
            white-space: pre;
        }

        .asm-content {
            white-space: pre;
        }

        .asm-line.clickable {
            cursor: pointer;
        }

        .asm-line.clickable:hover {
            background-color: var(--vscode-list-hoverBackground);
        }

        /* Filter classes */
        .asm-line.hidden {
            display: none;
        }

        /* Force visible for highlighted lines even when filtered */
        .asm-line.force-visible {
            display: block !important;
            opacity: 0.7;
            border-left: 3px solid var(--vscode-editorInfo-foreground);
        }

        .asm-line.filter-directive.hidden {
            display: none;
        }

        .asm-line.filter-comment.hidden {
            display: none;
        }

        .asm-line.filter-empty.hidden {
            display: none;
        }

        .asm-line.highlighted {
            background-color: var(--vscode-editor-selectionBackground);
        }

        .outdated-warning {
            background-color: var(--vscode-inputValidation-warningBackground);
            border: 1px solid var(--vscode-inputValidation-warningBorder);
            color: var(--vscode-inputValidation-warningForeground);
            padding: 8px 16px;
            margin: 0;
            font-size: 12px;
            font-weight: 500;
        }

        .outdated-kernel {
            /* opacity: 0.85; - Removed: Don't dim old kernels, just show warning banner */
        }

        /* GCN Assembly Syntax Highlighting */
        .comment {
            color: var(--vscode-descriptionForeground);
            font-style: italic;
        }
        .directive {
            color: var(--vscode-textLink-foreground);
            font-weight: bold;
        }
        .instruction {
            color: var(--vscode-textPreformat-foreground);
            font-weight: 500;
        }
        .register {
            color: var(--vscode-editor-foreground);
            opacity: 0.9;
        }
        .immediate {
            color: var(--vscode-charts-green);
        }
        .label {
            color: var(--vscode-textLink-activeForeground);
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="toolbar">
        <select id="irTypeSelector" class="kernel-selector" onchange="filterByIRType(this.value)" title="Filter by IR type">
            ${irTypeOptions}
        </select>
        <select id="kernelSelector" class="kernel-selector" onchange="jumpToKernel(this.value)" title="Select a kernel version to view">
            ${kernelOptions}
        </select>
        <select id="sourceFileSelector" class="kernel-selector" onchange="filterBySourceFile(this.value)" title="Filter by source file">
            ${sourceFileOptions}
        </select>
        <span style="flex: 1"></span>
        <button class="icon-btn" onclick="refreshCache()" title="Refresh cache - reload kernels from cache">
            <i class="codicon codicon-refresh"></i>
        </button>
        <button class="icon-btn" onclick="clearCache()" title="Clear cache - delete all cached kernels">
            <i class="codicon codicon-trash"></i>
        </button>
        <button class="icon-btn" onclick="compareKernels()" ${this._allAssemblies.length < 2 ? 'disabled title="Need at least 2 kernel versions to compare"' : 'title="Compare with another version - side-by-side diff"'}>
            <i class="codicon codicon-diff"></i>
        </button>
        <button class="icon-btn" id="searchBtn" onclick="toggleSearch()" title="Search in assembly (Ctrl+F)">
            <i class="codicon codicon-search"></i>
        </button>
        <button class="icon-btn" id="filterBtn" onclick="toggleFilterMenu(event)" title="Filter assembly lines - hide directives, comments, instructions">
            <i class="codicon codicon-filter"></i>
        </button>
        <div class="filter-menu" id="filterMenu" style="display: none;">
            <div class="filter-section">
                <div class="filter-section-title">General</div>
                <label><input type="checkbox" id="filterDirectives" onchange="applyFilters()"> Hide directives</label>
                <label><input type="checkbox" id="filterComments" onchange="applyFilters()"> Hide comments</label>
                <label><input type="checkbox" id="filterEmpty" onchange="applyFilters()"> Hide empty lines</label>
            </div>
            <div class="filter-section" id="instructionFilters">
                <div class="filter-section-title">Instructions</div>
                <div class="filter-search-container">
                    <input type="text" id="instrSearchBox" placeholder="Search instructions..." onkeyup="filterInstructionList()" />
                </div>
                <div class="filter-actions">
                    <button class="filter-action-btn" onclick="selectAllInstructions()">Select All</button>
                    <button class="filter-action-btn" onclick="selectNoneInstructions()">Select None</button>
                </div>
                <div id="instructionList">
                    <!-- Dynamic instruction filters will be inserted here -->
                </div>
            </div>
        </div>
        <div class="toolbar-separator"></div>
        <button class="icon-btn" id="prevHighlightBtn" onclick="navigateToPrevHighlight()" title="Previous highlighted block (when Python line maps to multiple assembly blocks)" disabled>
            <i class="codicon codicon-chevron-up"></i>
        </button>
        <button class="icon-btn" id="nextHighlightBtn" onclick="navigateToNextHighlight()" title="Next highlighted block (when Python line maps to multiple assembly blocks)" disabled>
            <i class="codicon codicon-chevron-down"></i>
        </button>
        <div class="toolbar-separator"></div>
        <button class="icon-btn" onclick="scrollToTop()" title="Jump to top of assembly">
            <i class="codicon codicon-arrow-up"></i>
        </button>
        <button class="icon-btn" onclick="scrollToBottom()" title="Jump to bottom of assembly">
            <i class="codicon codicon-arrow-down"></i>
        </button>
    </div>
    <div class="search-bar" id="searchBar" style="display: none;">
        <input type="text" id="searchInput" placeholder="Find in assembly..." />
        <button class="search-nav-btn" onclick="findPrevious()" title="Previous match (Shift+F3)">▲</button>
        <button class="search-nav-btn" onclick="findNext()" title="Next match (F3)">▼</button>
        <span id="searchResults" class="search-results"></span>
        <button class="search-close-btn" onclick="toggleSearch()">✕</button>
    </div>
    <div class="content">
        ${assembliesHtml}
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        document.querySelectorAll('.asm-line').forEach(line => {
            line.addEventListener('click', () => {
                // Remove previous highlights
                document.querySelectorAll('.asm-line.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });

                // Highlight clicked line
                line.classList.add('highlighted');

                // Send message to extension to highlight source
                const lineNum = parseInt(line.dataset.line);
                const kernelIndex = parseInt(line.dataset.kernelIndex);
                vscode.postMessage({
                    command: 'highlightSource',
                    line: lineNum,
                    kernelIndex: kernelIndex
                });
            });
        });

        function refreshCache() {
            vscode.postMessage({ command: 'refresh' });
        }

        function clearCache() {
            vscode.postMessage({ command: 'clearCache' });
        }

        function editLabel(cachePath, currentLabel) {
            vscode.postMessage({
                command: 'requestLabelEdit',
                cachePath: cachePath,
                currentLabel: currentLabel || undefined
            });
        }

        function filterByIRType(irType) {
            vscode.postMessage({
                command: 'filterByIRType',
                irType: irType
            });
        }

        function compareKernels() {
            vscode.postMessage({ command: 'compareKernels' });
        }

        function jumpToKernel(index) {
            // Send message to extension to show selected kernel
            vscode.postMessage({
                command: 'jumpToKernel',
                kernelIndex: parseInt(index)
            });
        }

        function filterBySourceFile(sourceFile) {
            // Send message to extension to reload with selected file
            vscode.postMessage({
                command: 'filterByFile',
                sourceFile: sourceFile
            });
        }

        function openCacheFile(filePath) {
            // Send message to extension to open the cache file
            vscode.postMessage({
                command: 'openCacheFile',
                filePath: filePath
            });
        }

        function toggleFilterMenu(event) {
            const menu = document.getElementById('filterMenu');
            if (menu.style.display === 'none') {
                // Position menu below and to the left of the button (so it doesn't go off-screen)
                const btn = event?.target?.closest('.icon-btn');
                if (btn) {
                    const rect = btn.getBoundingClientRect();
                    menu.style.top = (rect.bottom + 4) + 'px';

                    // Position menu so its RIGHT edge aligns with button's RIGHT edge
                    // This prevents it from going off-screen to the right
                    menu.style.left = 'auto';
                    menu.style.right = (window.innerWidth - rect.right) + 'px';
                }
                menu.style.display = 'block';
            } else {
                menu.style.display = 'none';
            }
        }

        // Build dynamic instruction filters on page load
        function buildInstructionFilters() {
            const instructionCounts = {};

            // Scan all lines for instructions
            document.querySelectorAll('.asm-line').forEach(line => {
                line.classList.forEach(cls => {
                    if (cls.startsWith('instr-')) {
                        const instr = cls.substring(6); // Remove 'instr-' prefix
                        instructionCounts[instr] = (instructionCounts[instr] || 0) + 1;
                    }
                });
            });

            // Sort instructions alphabetically
            const sortedInstructions = Object.keys(instructionCounts).sort();

            // Populate the instruction list
            const instructionListDiv = document.getElementById('instructionList');
            if (instructionListDiv && sortedInstructions.length > 0) {
                let html = '';
                sortedInstructions.forEach(instr => {
                    const count = instructionCounts[instr];
                    const id = 'filter-instr-' + instr.replace(/[^a-z0-9]/g, '_');
                    html += '<label data-instr-name="' + instr.toLowerCase() + '"><input type="checkbox" id="' + id + '" data-instr="' + instr + '" onchange="applyFilters()"> ' + instr + ' (' + count + ')</label>';
                });
                instructionListDiv.innerHTML = html;
            }

            // Update the section title with count
            const titleDiv = document.querySelector('#instructionFilters .filter-section-title');
            if (titleDiv) {
                titleDiv.textContent = 'Instructions (' + sortedInstructions.length + ')';
            }
        }

        // Filter instruction list based on search
        function filterInstructionList() {
            const searchText = document.getElementById('instrSearchBox').value.toLowerCase();
            const labels = document.querySelectorAll('#instructionList label');

            labels.forEach(label => {
                const instrName = label.getAttribute('data-instr-name');
                if (instrName.includes(searchText)) {
                    label.classList.remove('filter-hidden');
                } else {
                    label.classList.add('filter-hidden');
                }
            });
        }

        // Select all visible instructions
        function selectAllInstructions() {
            const checkboxes = document.querySelectorAll('#instructionList label:not(.filter-hidden) input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = true;
            });
            applyFilters();
        }

        // Select none (clear all)
        function selectNoneInstructions() {
            const checkboxes = document.querySelectorAll('#instructionList input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = false;
            });
            applyFilters();
        }

        function applyFilters() {
            const hideDirectives = document.getElementById('filterDirectives').checked;
            const hideComments = document.getElementById('filterComments').checked;
            const hideEmpty = document.getElementById('filterEmpty').checked;

            // Get all instruction filter checkboxes
            const instrCheckboxes = document.querySelectorAll('#instructionList input[type="checkbox"]');
            const hiddenInstructions = new Set();
            instrCheckboxes.forEach(cb => {
                if (cb.checked) {
                    hiddenInstructions.add('instr-' + cb.getAttribute('data-instr'));
                }
            });

            let visibleLineNum = 1;
            document.querySelectorAll('.asm-line').forEach(line => {
                line.classList.remove('hidden');

                let isHidden = false;
                if (hideDirectives && line.classList.contains('filter-directive')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }
                if (hideComments && line.classList.contains('filter-comment')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }
                if (hideEmpty && line.classList.contains('filter-empty')) {
                    line.classList.add('hidden');
                    isHidden = true;
                }

                // Check if line has any hidden instruction
                if (!isHidden) {
                    line.classList.forEach(cls => {
                        if (hiddenInstructions.has(cls)) {
                            line.classList.add('hidden');
                            isHidden = true;
                        }
                    });
                }

                // Renumber visible lines
                const lineNumSpan = line.querySelector('.line-number');
                if (lineNumSpan) {
                    if (isHidden) {
                        lineNumSpan.textContent = '';
                    } else {
                        lineNumSpan.textContent = visibleLineNum.toString().padStart(4, ' ');
                        visibleLineNum++;
                    }
                }
            });

            // Save filter state for diff view to inherit
            const hiddenInstrArray = [];
            instrCheckboxes.forEach(cb => {
                if (cb.checked) {
                    hiddenInstrArray.push(cb.getAttribute('data-instr'));
                }
            });

            vscode.postMessage({
                command: 'saveFilterState',
                filterState: {
                    hideDirectives: hideDirectives,
                    hideComments: hideComments,
                    hideEmpty: hideEmpty,
                    hiddenInstructions: hiddenInstrArray
                }
            });
        }

        // Initialize instruction filters after DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            buildInstructionFilters();
            // Save initial filter state (all unchecked) for diff view to inherit
            setTimeout(() => {
                applyFilters(); // This will save the initial state
            }, 150);
        });
        // Also build after a short delay to ensure content is loaded
        setTimeout(() => {
            buildInstructionFilters();
            // Save initial filter state
            setTimeout(() => {
                applyFilters();
            }, 150);
        }, 100);

        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function scrollToBottom() {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }

        function navigateToNextHighlight() {
            if (highlightedBlocks.length === 0) return;
            currentHighlightIndex = (currentHighlightIndex + 1) % highlightedBlocks.length;
            highlightedBlocks[currentHighlightIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        function navigateToPrevHighlight() {
            if (highlightedBlocks.length === 0) return;
            currentHighlightIndex = (currentHighlightIndex - 1 + highlightedBlocks.length) % highlightedBlocks.length;
            highlightedBlocks[currentHighlightIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Search functionality
        let searchMatches = [];
        let currentMatchIndex = -1;

        function toggleSearch() {
            const searchBar = document.getElementById('searchBar');
            const searchInput = document.getElementById('searchInput');

            if (searchBar.style.display === 'none') {
                searchBar.style.display = 'flex';
                searchInput.focus();
                searchInput.select();
            } else {
                searchBar.style.display = 'none';
                clearSearch();
            }
        }

        function clearSearch() {
            document.querySelectorAll('.search-match, .search-match-current').forEach(el => {
                el.classList.remove('search-match', 'search-match-current');
            });
            searchMatches = [];
            currentMatchIndex = -1;
            document.getElementById('searchResults').textContent = '';
        }

        function performSearch() {
            const searchText = document.getElementById('searchInput').value;
            clearSearch();

            if (!searchText) return;

            const lines = document.querySelectorAll('.asm-line .asm-content');
            lines.forEach((line, index) => {
                const text = line.textContent;
                if (text.toLowerCase().includes(searchText.toLowerCase())) {
                    line.parentElement.classList.add('search-match');
                    searchMatches.push(line.parentElement);
                }
            });

            if (searchMatches.length > 0) {
                currentMatchIndex = 0;
                highlightCurrentMatch();
                updateSearchResults();
            } else {
                document.getElementById('searchResults').textContent = 'No results';
            }
        }

        function highlightCurrentMatch() {
            searchMatches.forEach((match, index) => {
                if (index === currentMatchIndex) {
                    match.classList.add('search-match-current');
                    match.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    match.classList.remove('search-match-current');
                }
            });
        }

        function updateSearchResults() {
            if (searchMatches.length > 0) {
                document.getElementById('searchResults').textContent =
                    (currentMatchIndex + 1) + ' of ' + searchMatches.length;
            }
        }

        function findNext() {
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
            highlightCurrentMatch();
            updateSearchResults();
        }

        function findPrevious() {
            if (searchMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
            highlightCurrentMatch();
            updateSearchResults();
        }

        // Search input handler
        document.addEventListener('DOMContentLoaded', () => {
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.addEventListener('input', performSearch);
                searchInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        if (e.shiftKey) {
                            findPrevious();
                        } else {
                            findNext();
                        }
                    } else if (e.key === 'Escape') {
                        toggleSearch();
                    }
                });
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                toggleSearch();
            } else if (e.key === 'F3') {
                e.preventDefault();
                if (e.shiftKey) {
                    findPrevious();
                } else {
                    findNext();
                }
            }
        });

        // Close filter menu when clicking outside
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('filterMenu');
            const filterBtn = document.getElementById('filterBtn');

            if (menu && filterBtn && !menu.contains(e.target) && !filterBtn.contains(e.target)) {
                menu.style.display = 'none';
            }
        });

        // Listen for messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            console.log('Webview received message:', message.command);

            if (message.command === 'updateFileDropdown') {
                console.log('Updating file dropdown with', message.files.length, 'files');

                // Find the source file dropdown by ID
                const sourceFileSelect = document.getElementById('sourceFileSelector');
                if (sourceFileSelect) {
                    // Build new options (no placeholder)
                    const options = [];
                    message.files.forEach(file => {
                        const selected = file.path === message.selectedFile ? ' selected' : '';
                        options.push(\`<option value="\${file.path}"\${selected}>\${file.basename}</option>\`);
                    });

                    sourceFileSelect.innerHTML = options.join('');
                    console.log('File dropdown updated with', message.files.length, 'files');
                } else {
                    console.log('Source file dropdown not found');
                }
            } else if (message.command === 'highlightAssembly') {
                // Clear previous highlights
                document.querySelectorAll('.asm-line.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });

                // Clear previous navigation blocks
                highlightedBlocks = [];
                currentHighlightIndex = -1;

                // Collect all highlighted lines first
                let allHighlightedLines = [];
                let firstVisibleLine = null;
                message.lines.forEach(lineNum => {
                    const line = document.querySelector(\`.asm-line[data-line="\${lineNum}"]\`);
                    if (line) {
                        // Don't highlight directives, comments, or empty lines
                        const isDirective = line.classList.contains('filter-directive');
                        const isComment = line.classList.contains('filter-comment');
                        const isEmpty = line.classList.contains('filter-empty');

                        if (isDirective || isComment || isEmpty) {
                            return; // Skip this line entirely
                        }

                        // Don't highlight lines that are hidden by instruction filters
                        if (line.classList.contains('hidden')) {
                            return; // Skip hidden instructions
                        }

                        line.classList.add('highlighted');
                        allHighlightedLines.push({ element: line, lineNum: lineNum });

                        // Track first visible/highlighted line for scrolling
                        if (!firstVisibleLine) {
                            firstVisibleLine = line;
                        }
                    }
                });

                // Group consecutive lines into blocks
                if (allHighlightedLines.length > 0) {
                    let currentBlock = [allHighlightedLines[0]];

                    for (let i = 1; i < allHighlightedLines.length; i++) {
                        const prev = allHighlightedLines[i - 1];
                        const curr = allHighlightedLines[i];

                        // If consecutive (within 1 line), add to current block
                        if (curr.lineNum - prev.lineNum <= 1) {
                            currentBlock.push(curr);
                        } else {
                            // Start a new block
                            highlightedBlocks.push(currentBlock[0].element); // Store first line of block
                            currentBlock = [curr];
                        }
                    }
                    // Don't forget the last block
                    highlightedBlocks.push(currentBlock[0].element);
                }

                // Enable/disable navigation buttons based on block count
                const prevBtn = document.getElementById('prevHighlightBtn');
                const nextBtn = document.getElementById('nextHighlightBtn');
                if (highlightedBlocks.length > 1) {
                    currentHighlightIndex = 0;
                    prevBtn.disabled = false;
                    nextBtn.disabled = false;
                } else {
                    prevBtn.disabled = true;
                    nextBtn.disabled = true;
                }

                // Smart scroll: only scroll if the first highlighted line is not currently visible
                if (firstVisibleLine) {
                    const rect = firstVisibleLine.getBoundingClientRect();
                    const viewportHeight = window.innerHeight;

                    // Check if the line is outside the viewport
                    const isAboveViewport = rect.top < 0;
                    const isBelowViewport = rect.bottom > viewportHeight;

                    if (isAboveViewport || isBelowViewport) {
                        // Line is not visible, scroll to it
                    firstVisibleLine.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    // Otherwise, line is already visible - don't scroll, keep user's position
                }
            } else if (message.command === 'scrollToLine') {
                // Scroll to a specific assembly line
                const line = document.querySelector(\`.asm-line[data-line="\${message.line}"]\`);
                if (line) {
                    line.scrollIntoView({ behavior: 'smooth', block: 'center' });

                    // Briefly highlight it
                    line.classList.add('highlighted');
                    setTimeout(() => {
                        line.classList.remove('highlighted');
                    }, 1500);
                }
            } else if (message.command === 'scrollToTop') {
                // Scroll assembly view to top
                const firstLine = document.querySelector('.asm-line');
                if (firstLine) {
                    firstLine.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            } else if (message.command === 'restoreFilters') {
                console.log('Restoring filter state:', message.filterState);

                // Function to restore filter state
                function restoreFilterState() {
                    const filterState = message.filterState;
                    if (!filterState) return;

                    // Restore general filters
                    if (filterState.hideDirectives) {
                        const el = document.getElementById('filterDirectives');
                        if (el) el.checked = true;
                    }
                    if (filterState.hideComments) {
                        const el = document.getElementById('filterComments');
                        if (el) el.checked = true;
                    }
                    if (filterState.hideEmpty) {
                        const el = document.getElementById('filterEmpty');
                        if (el) el.checked = true;
                    }

                    // Restore instruction filters (wait for them to be built)
                    if (filterState.hiddenInstructions && filterState.hiddenInstructions.length > 0) {
                        let retries = 0;
                        const restoreInstrFilters = () => {
                            let restored = 0;
                            filterState.hiddenInstructions.forEach(instr => {
                                const id = 'filter-instr-' + instr.replace(/[^a-z0-9]/g, '_');
                                const checkbox = document.getElementById(id);
                                if (checkbox) {
                                    checkbox.checked = true;
                                    restored++;
                                }
                            });

                            console.log('Restored ' + restored + ' instruction filters out of ' + filterState.hiddenInstructions.length);

                            // Retry if not all filters were restored
                            if (restored < filterState.hiddenInstructions.length && retries < 10) {
                                retries++;
                                setTimeout(restoreInstrFilters, 100);
                            } else {
                                // Apply filters once all checkboxes are restored
                                applyFilters();
                            }
                        };

                        setTimeout(restoreInstrFilters, 150);
                    } else {
                        // Apply general filters immediately if no instruction filters
                        applyFilters();
                    }
                }

                // Wait for instruction list to be built
                setTimeout(restoreFilterState, 250);
            }
        });
    </script>
</body>
</html>`;
    }

    private getNoAssemblyHtml(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>No Assembly Found</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 40px;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            text-align: center;
        }
        .message {
            max-width: 600px;
            margin: 0 auto;
        }
        h2 {
            color: var(--vscode-editorWarning-foreground);
        }
        .cache-path {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 8px;
            border-radius: 4px;
            margin: 20px 0;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="message">
        <h2>⚠️ No Assembly Found</h2>
        <p>No cached Triton kernels were found in the cache directory.</p>
        <div class="cache-path">${this._compiler.getCachePath()}</div>
        <p>To generate assembly:</p>
        <ol style="text-align: left; display: inline-block;">
            <li>Execute your Triton kernel code</li>
            <li>Click the refresh button to reload</li>
        </ol>
    </div>
</body>
</html>`;
    }

    private getErrorHtml(error: string): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Error</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 40px;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-errorForeground);
        }
    </style>
</head>
<body>
    <h2>Error Loading Assembly</h2>
    <pre>${this.escapeHtml(error)}</pre>
</body>
</html>`;
    }

    private escapeHtml(text: string): string {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    private formatTimestamp(date: Date): string {
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        // Show relative time for recent builds
        if (diffMins < 1) {
            return 'just now';
        } else if (diffMins < 60) {
            return `${diffMins}m ago`;
        } else if (diffHours < 24) {
            return `${diffHours}h ago`;
        } else if (diffDays < 7) {
            return `${diffDays}d ago`;
        } else {
            // Show full date for older builds
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
    }

    private highlightGCNSyntax(line: string): string {
        let escaped = this.escapeHtml(line);

        // Comments (anything after // or ; or #)
        escaped = escaped.replace(/(\/\/|;|#)(.*)$/, '<span class="comment">$1$2</span>');

        // Directives (start with .)
        escaped = escaped.replace(/^(\s*)(\.\w+)/g, '$1<span class="directive">$2</span>');

        // Instructions (v_ for VALU, s_ for SALU, ds_ for LDS, etc.)
        escaped = escaped.replace(/\b([vs]_\w+|ds_\w+|buffer_\w+|flat_\w+|tbuffer_\w+|image_\w+)\b/g,
            '<span class="instruction">$1</span>');

        // Registers (v[num], s[num], vcc, exec, etc.)
        escaped = escaped.replace(/\b(v\[\d+:\d+\]|v\[\d+\]|s\[\d+:\d+\]|s\[\d+\]|vcc|exec|m0|tma|tba|flat_scratch)\b/g,
            '<span class="register">$1</span>');

        // Immediate values (hex and decimal)
        escaped = escaped.replace(/\b(0x[0-9a-fA-F]+|\d+)\b/g, '<span class="immediate">$1</span>');

        // Labels (word followed by :)
        escaped = escaped.replace(/^(\s*)(\w+):/g, '$1<span class="label">$2</span>:');

        return escaped;
    }

    public dispose() {
        AssemblyViewerPanel.currentPanel = undefined;

        // Clear any pending theme change timeout
        if (this._themeChangeTimeout) {
            clearTimeout(this._themeChangeTimeout);
        }

        // Clear all source code highlights when closing viewer
        this._highlightManager.clearHighlights();

        // Clear saved state when panel is closed
        if (this._context) {
            this._context.workspaceState.update('assemblyViewerState', undefined);
        }

        this._panel.dispose();

        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }
}

