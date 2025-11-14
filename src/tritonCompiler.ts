import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { promisify } from 'util';
import { logger } from './logger';

const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const stat = promisify(fs.stat);
const unlink = promisify(fs.unlink);

export type IRType = 'gcn' | 'llvm' | 'ttgir' | 'ttir' | 'unknown';

export interface CallsiteInfo {
    // Map from callsite location (file:line) to array of inlined locations (file:line)
    // e.g., "gemm.py:193" -> ["iris.py:1706", "iris.py:1708"]
    callsiteMap: Map<string, string[]>;
}

export interface KernelAssembly {
    kernelName: string;
    assembly: string;
    cachePath: string;
    timestamp: Date;
    sourceFiles: string[];  // All source files from .file directives
    irType: IRType;  // Type of IR (GCN, LLVM, TTGIR, TTIR, etc.)
    callsites?: CallsiteInfo;  // Optional callsite information from TTIR/TTGIR
}

export class TritonCompiler {
    private tritonCachePath: string = '';

    constructor() {
        this.updateCachePath();
    }

    /**
     * Update cache path from configuration
     */
    private updateCachePath(): void {
        const config = vscode.workspace.getConfiguration('trident');
        let cachePath = config.get<string>('cacheDirectory', '~/.triton/cache');

        // Expand ~ to home directory
        if (cachePath.startsWith('~')) {
            const homeDir = process.env.HOME || process.env.USERPROFILE || '';
            cachePath = path.join(homeDir, cachePath.slice(1));
        }

        this.tritonCachePath = cachePath;
    }

    /**
     * Get the current cache path (useful for displaying to user)
     */
    public getCachePath(): string {
        return this.tritonCachePath;
    }

    /**
     * Clear the entire Triton cache directory
     */
    public async clearCache(): Promise<void> {
        // Update cache path from configuration in case it changed
        this.updateCachePath();

        if (!fs.existsSync(this.tritonCachePath)) {
            throw new Error(`Triton cache directory not found: ${this.tritonCachePath}`);
        }

        const entries = await readdir(this.tritonCachePath);

        for (const entry of entries) {
            const entryPath = path.join(this.tritonCachePath, entry);
            const entryStat = await stat(entryPath);

            if (entryStat.isDirectory()) {
                // Recursively delete directory
                await this.deleteDirectory(entryPath);
            } else {
                // Delete file
                await unlink(entryPath);
            }
        }
    }

    /**
     * Recursively delete a directory and all its contents
     */
    private async deleteDirectory(dirPath: string): Promise<void> {
        const entries = await readdir(dirPath);

        for (const entry of entries) {
            const entryPath = path.join(dirPath, entry);
            const entryStat = await stat(entryPath);

            if (entryStat.isDirectory()) {
                await this.deleteDirectory(entryPath);
            } else {
                await unlink(entryPath);
            }
        }

        // Remove the directory itself (use fs.promises for better compatibility)
        await fs.promises.rmdir(dirPath);
    }

    /**
     * Extract all source file paths from IR content (supports multiple IR types)
     */
    extractSourceFiles(assembly: string, irType: IRType): string[] {
        const sourceFiles: string[] = [];

        if (irType === 'gcn') {
            // GCN assembly uses .file directives
            // Look for .file directives with two formats:
            // Format 1: .file 1 "/path/to/source.py"
            // Format 2: .file 1 ";/path/to/dir" "filename.py"
            const lines = assembly.split('\n');
            for (const line of lines) {
                // Format 2: directory and filename separate
                const format2Match = line.match(/\.file\s+\d+\s+"([^"]+)"\s+"([^"]+)"/);
                if (format2Match) {
                    const directory = format2Match[1].replace(/^;/, ''); // Remove leading semicolon
                    const filename = format2Match[2];
                    const fullPath = path.join(directory, filename);
                    if (!sourceFiles.includes(fullPath)) {
                        sourceFiles.push(fullPath);
                    }
                    continue;
                }

                // Format 1: single full path
                const format1Match = line.match(/\.file\s+\d+\s+"([^"]+)"/);
                if (format1Match) {
                    const filePath = format1Match[1];
                    if (!sourceFiles.includes(filePath)) {
                        sourceFiles.push(filePath);
                    }
                }
            }
        } else if (irType === 'llvm') {
            // LLVM IR uses !DIFile debug metadata
            // Pattern: !DIFile(filename: "file.py", directory: "/path/to")
            const difilePattern = /!DIFile\([^)]*filename:\s*"([^"]+)"[^)]*directory:\s*"([^"]+)"[^)]*\)/g;
            const matches = Array.from(assembly.matchAll(difilePattern));
            for (const match of matches) {
                const filename = match[1];
                const directory = match[2];
                const filePath = path.join(directory, filename);
                if (!sourceFiles.includes(filePath)) {
                    sourceFiles.push(filePath);
                }
            }
        } else if (irType === 'ttgir' || irType === 'ttir') {
            // TTGIR and TTIR use #loc = loc("filepath":line:col) format (note: colon is AFTER the quote)
            // Pattern: #loc = loc("filepath":line:col)
            const locPattern = /#loc\s*=\s*loc\("([^"]+)":\d+:\d+\)/g;
            const matches = Array.from(assembly.matchAll(locPattern));

            for (const match of matches) {
                const filePath = match[1];
                if (!sourceFiles.includes(filePath)) {
                    sourceFiles.push(filePath);
                }
            }

            // Also check for loc("filepath":line:col) in function definitions
            const locPattern2 = /loc\("([^"]+)":\d+:\d+\)/g;
            const matches2 = Array.from(assembly.matchAll(locPattern2));

            for (const match of matches2) {
                const filePath = match[1];
                if (!sourceFiles.includes(filePath)) {
                    sourceFiles.push(filePath);
                }
            }

            logger.log(`[Trident] ${irType.toUpperCase()} file extracted ${sourceFiles.length} source files: ${sourceFiles.join(', ')}`);
        }
        // Unknown types - no source file extraction for now

        return sourceFiles;
    }

    /**
     * Extract kernel names from Python source code
     */
    extractKernelNames(pythonSource: string): string[] {
        const kernelNames: string[] = [];

        // Match @triton.jit decorated functions (handle multi-line)
        // Look for @triton.jit (with possible arguments), then find the next def statement
        const lines = pythonSource.split('\n');
        let nextLineIsKernel = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            // Check for @triton.jit decorator
            if (line.startsWith('@triton.jit') || line.startsWith('@jit')) {
                nextLineIsKernel = true;
                continue;
            }

            // If previous line was @triton.jit, look for def
            if (nextLineIsKernel && line.startsWith('def ')) {
                const match = line.match(/def\s+(\w+)\s*\(/);
                if (match) {
                    kernelNames.push(match[1]);
                    logger.log(`[Trident] Found kernel: ${match[1]}`);
                }
                nextLineIsKernel = false;
            } else if (nextLineIsKernel && !line.startsWith('@')) {
                // Reset if we hit a non-decorator line without def
                nextLineIsKernel = false;
            }
        }

        return kernelNames;
    }

    /**
     * Detect which kernel the cursor is in based on line number
     * Returns the kernel name if found, undefined otherwise
     */
    detectKernelAtLine(pythonSource: string, lineNumber: number): string | undefined {
        const lines = pythonSource.split('\n');

        // Find all kernel definitions with their line ranges
        const kernelRanges: Array<{ name: string, startLine: number, endLine: number }> = [];
        let nextLineIsKernel = false;
        let currentKernelStart = -1;
        let currentKernelName = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            // Check for @triton.jit decorator
            if (line.startsWith('@triton.jit') || line.startsWith('@jit')) {
                nextLineIsKernel = true;
                continue;
            }

            // If previous line was @triton.jit, look for def
            if (nextLineIsKernel && line.startsWith('def ')) {
                const match = line.match(/def\s+(\w+)\s*\(/);
                if (match) {
                    // End previous kernel if there was one
                    if (currentKernelStart >= 0) {
                        kernelRanges.push({
                            name: currentKernelName,
                            startLine: currentKernelStart,
                            endLine: i - 1
                        });
                    }
                    // Start new kernel
                    currentKernelName = match[1];
                    currentKernelStart = i;
                }
                nextLineIsKernel = false;
            } else if (nextLineIsKernel && !line.startsWith('@')) {
                nextLineIsKernel = false;
            }
        }

        // Add the last kernel
        if (currentKernelStart >= 0) {
            kernelRanges.push({
                name: currentKernelName,
                startLine: currentKernelStart,
                endLine: lines.length - 1
            });
        }

        // Find which kernel the line is in
        for (const kernel of kernelRanges) {
            if (lineNumber >= kernel.startLine && lineNumber <= kernel.endLine) {
                logger.log(`[Trident] Line ${lineNumber} is in kernel: ${kernel.name}`);
                return kernel.name;
            }
        }

        logger.log(`[Trident] Line ${lineNumber} is not in any kernel`);
        return undefined;
    }

    /**
     * Search the Triton cache for assemblies matching the given kernel names
     * Groups all IR types (GCN, TTGIR, TTIR, LLVM) together per kernel
     */
    async findCachedAssemblies(kernelNames: Set<string>, _workspaceFolders?: string[]): Promise<KernelAssembly[]> {
        // Update cache path from configuration in case it changed
        this.updateCachePath();

        const assemblies: KernelAssembly[] = [];

        try {
            if (!fs.existsSync(this.tritonCachePath)) {
                throw new Error(`Triton cache directory not found: ${this.tritonCachePath}`);
            }

            const entries = await readdir(this.tritonCachePath);

            for (const entry of entries) {
                const entryPath = path.join(this.tritonCachePath, entry);
                const entryStat = await stat(entryPath);

                if (entryStat.isDirectory()) {
                    // Search for all IR files in this cache directory
                    const irFiles = await this.findIRFiles(entryPath);

                    // Extract kernel name from GCN file (most reliable)
                    let kernelName: string | undefined;
                    for (const irFile of irFiles) {
                        if (irFile.irType === 'gcn') {
                            try {
                                const content = await readFile(irFile.path, 'utf-8');
                                kernelName = this.extractKernelName(irFile.path, content);
                                break;
                            } catch (error) {
                                logger.log(`[Trident] Error reading GCN file ${irFile.path}: ${error}`);
                            }
                        }
                    }

                    // If no GCN, try any IR file
                    if (!kernelName) {
                        for (const irFile of irFiles) {
                            try {
                                const content = await readFile(irFile.path, 'utf-8');
                                kernelName = this.extractKernelName(irFile.path, content);
                                break;
                            } catch (error) {
                                // Continue
                            }
                        }
                    }

                    // Skip if kernel name doesn't match what we're looking for
                    if (!kernelName || !kernelNames.has(kernelName)) {
                        continue;
                    }

                    // Process all IR files for this kernel - they're all in the same directory
                    for (const irFile of irFiles) {
                        const content = await readFile(irFile.path, 'utf-8');
                        const sourceFiles = this.extractSourceFiles(content, irFile.irType);

                        // Parse callsite info for TTIR/TTGIR
                        let callsites: CallsiteInfo | undefined;
                        if (irFile.irType === 'ttir' || irFile.irType === 'ttgir') {
                            callsites = this.parseCallsites(content);
                        }

                        assemblies.push({
                            kernelName: kernelName,
                            assembly: content,
                            cachePath: irFile.path,
                            timestamp: entryStat.mtime,
                            sourceFiles,
                            irType: irFile.irType,
                            callsites
                        });
                    }
                }
            }

            // Sort by kernel name, then by timestamp (most recent first)
            assemblies.sort((a, b) => {
                if (a.kernelName !== b.kernelName) {
                    return a.kernelName.localeCompare(b.kernelName);
                }
                return b.timestamp.getTime() - a.timestamp.getTime();
            });

            // Debug: Log what we found
            logger.log(`[Trident] ===== Found ${assemblies.length} assemblies for ${kernelNames.size} kernels =====`);
            const byKernel = new Map<string, number>();
            assemblies.forEach(asm => {
                byKernel.set(asm.kernelName, (byKernel.get(asm.kernelName) || 0) + 1);
            });
            logger.log(`[Trident] Kernels found: ${Array.from(byKernel.entries()).map(([k, v]) => `${k} (${v} IRs)`).join(', ')}`);

            return assemblies;

        } catch (error) {
            console.error('Error searching Triton cache:', error);
            throw error;
        }
    }

    /**
     * Find all IR files in a directory (GCN, LLVM, TTGIR, TTIR, etc.)
     */
    private async findIRFiles(dir: string): Promise<Array<{ path: string, irType: IRType }>> {
        const irFiles: Array<{ path: string, irType: IRType }> = [];

        try {
            const files = await readdir(dir);
            logger.log(`[Trident] ===== Scanning directory: ${dir} =====`);
            logger.log(`[Trident] Found ${files.length} files/directories`);

            for (const file of files) {
                const filePath = path.join(dir, file);
                const fileStat = await stat(filePath);

                if (fileStat.isFile()) {
                    // Skip JSON, binary, and source files explicitly
                    const lower = file.toLowerCase();
                    if (lower.endsWith('.json') || lower.endsWith('.hsaco') || lower.endsWith('.so') ||
                        lower.endsWith('.py') || lower.endsWith('.pyi') || lower.endsWith('.c') ||
                        lower.endsWith('.cpp') || lower.endsWith('.h') || lower.endsWith('.hpp')) {
                        logger.log(`[Trident] Skipping non-IR file: ${file}`);
                        continue;
                    }

                    // Try to detect IR type by filename patterns first
                    let irType: IRType | undefined = this.detectIRTypeFromFilename(file);
                    logger.log(`[Trident] File: ${file}, filename detection: ${irType || 'none'}`);

                    // If not detected by filename (or ambiguous like .s/.asm), check content
                    if (!irType) {
                        try {
                            const content = await readFile(filePath, 'utf-8');
                            // Skip empty or very small files
                            if (content.trim().length < 10) {
                                logger.log(`[Trident] Skipping small/empty file: ${file} (${content.trim().length} chars)`);
                                continue;
                            }
                            irType = this.detectIRTypeFromContent(content, file);
                            logger.log(`[Trident] File: ${file} -> Detected IR type: ${irType} (from content, ${content.length} chars)`);
                        } catch (error) {
                            logger.log(`[Trident] Error reading file ${file}: ${error}`);
                            // Skip files that can't be read
                            continue;
                        }
                    } else {
                        logger.log(`[Trident] File: ${file} -> Detected IR type: ${irType} (from filename)`);
                    }

                    // Include all detected IR types (including 'unknown' for now to see what we're missing)
                    if (irType) {
                        irFiles.push({ path: filePath, irType });
                        logger.log(`[Trident] ✓ Added ${file} as ${irType}`);
                    } else {
                        logger.log(`[Trident] ✗ File: ${file} -> No IR type detected, skipping`);
                    }
                } else {
                    logger.log(`[Trident] Skipping non-file: ${file}`);
                }
            }

            logger.log(`[Trident] ===== Total IR files found in ${dir}: ${irFiles.length} =====`);
            const irTypeCounts = new Map<IRType, number>();
            irFiles.forEach(f => {
                irTypeCounts.set(f.irType, (irTypeCounts.get(f.irType) || 0) + 1);
            });
            logger.log(`[Trident] IR type breakdown: ${JSON.stringify(Object.fromEntries(irTypeCounts))}`);
            irFiles.forEach(f => {
                logger.log(`[Trident]   - ${path.basename(f.path)}: ${f.irType}`);
            });
        } catch (error) {
            console.error(`Error reading directory ${dir}:`, error);
        }

        return irFiles;
    }

    /**
     * Detect IR type from filename patterns
     * Only checks for known IR file extensions - ignores everything else
     */
    private detectIRTypeFromFilename(filename: string): IRType | undefined {
        const lower = filename.toLowerCase();

        // Only check for known IR file extensions
        // GCN files
        if (lower.endsWith('.amdgcn') || lower.endsWith('.s') || lower.endsWith('.asm')) {
            // .s and .asm could be GCN, but need content check to be sure
            if (lower.endsWith('.amdgcn') || lower.includes('amdgcn')) {
                return 'gcn';
            }
            // .s/.asm need content check - return undefined to trigger content detection
            return undefined;
        }

        // LLVM files
        if (lower.endsWith('.ll') || lower.endsWith('.llir')) {
            return 'llvm';
        }

        // TTGIR files
        if (lower.endsWith('.ttgir')) {
            return 'ttgir';
        }

        // TTIR files
        if (lower.endsWith('.ttir')) {
            return 'ttir';
        }

        // Unknown extension - not an IR file we care about
        return undefined;
    }

    /**
     * Detect IR type from file content
     */
    private detectIRTypeFromContent(content: string, _filename: string): IRType {

        // Check for TTIR first (earlier stage, uses module { syntax)
        // TTIR has module { at the start, while TTGIR has module attributes { with #ttg.* attributes
        if (content.includes('module {') && content.includes('tt.func')) {
            // Check if it has TTGIR-specific attributes (distinguishes from TTIR)
            if (content.includes('#ttg.') || content.includes('ttg.target')) {
                return 'ttgir';
            }
            return 'ttir';
        }

        // Check for TTGIR (has #ttg.* attributes or ttg.target)
        if (content.includes('#ttg.') || content.includes('ttg.target') ||
            (content.includes('tt.func') && (content.includes('module attributes') || content.includes('#blocked')))) {
            return 'ttgir';
        }

        // Check for GCN assembly markers (AMD-specific)
        const gcnMarkers = [
            '.amdgcn', '.amdgpu', 's_load_', 'v_mov_', 'v_add_', 's_waitcnt',
            '.hsa_code_object', 'amdgpu_hsa_kernel', 'gfx', '.amdhsa_',
            's_mov_b32', 'v_mov_b32', 's_mov_b64', 'v_cmp_', 's_cmp_'
        ];
        if (gcnMarkers.some(marker => content.includes(marker))) {
            return 'gcn';
        }

        // Check for LLVM IR markers (more lenient)
        // LLVM IR has very distinctive syntax
        const hasLLVMSyntax = (
            (content.includes('define') || content.includes('declare')) &&
            (content.includes('@') || content.includes('!')) &&
            (content.includes('=') || content.includes('call ') || content.includes('ret ') || content.includes('alloca'))
        );
        if (hasLLVMSyntax) {
            return 'llvm';
        }

        // Check if it looks like assembly but we can't determine type
        // Look for common assembly patterns
        const assemblyPatterns = [
            /^\s*\.(text|data|section|globl|global|align|file|loc)/m,
            /^\s*[a-zA-Z_][a-zA-Z0-9_]*:/m,  // Label
            /^\s*[a-zA-Z][a-zA-Z0-9_]*\s+/m   // Instruction
        ];
        const looksLikeAssembly = assemblyPatterns.some(pattern => pattern.test(content));

        if (looksLikeAssembly) {
            // Default to GCN if it looks like assembly but we can't determine
            // (since this is an AMD-focused extension)
            return 'gcn';
        }

        // Default to unknown if we can't determine
        return 'unknown';
    }

    /**
     * Extract kernel name from file path or content (works for multiple IR types)
     */
    private extractKernelName(filePath: string, content: string): string {
        // For GCN files, prioritize .globl directive (most reliable)
        // Look for .globl followed by a label with the same name (the actual kernel function)
        const globalMatches = Array.from(content.matchAll(/\.globl\s+(\w+)/g));
        for (const match of globalMatches) {
            const name = match[1];
            // Check if this name appears as a label (name followed by colon, possibly with whitespace)
            // This indicates it's the actual function entry point
            const labelPattern = new RegExp(`^\\s*${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}:`, 'm');
            if (labelPattern.test(content)) {
                logger.log(`[Trident] Extracted kernel name from .globl (with matching label): ${name}`);
                return name;
            }
        }
        // If no matching label found, use the first .globl (fallback)
        if (globalMatches.length > 0) {
            const name = globalMatches[0][1];
            logger.log(`[Trident] Extracted kernel name from .globl (first match): ${name}`);
            return name;
        }

        // Try GCN-specific patterns
        const gcnKernelMatch = content.match(/\.amdhsa_kernel\s+(\w+)/);
        if (gcnKernelMatch) {
            logger.log(`[Trident] Extracted kernel name from .amdhsa_kernel: ${gcnKernelMatch[1]}`);
            return gcnKernelMatch[1];
        }

        // Try LLVM patterns
        const llvmDefineMatch = content.match(/define\s+(?:[^@]*@)?(\w+)\s*\(/);
        if (llvmDefineMatch) {
            logger.log(`[Trident] Extracted kernel name from LLVM define: ${llvmDefineMatch[1]}`);
            return llvmDefineMatch[1];
        }

        // Try TTGIR patterns
        const ttgirFuncMatch = content.match(/tt\.func\s+@(\w+)/);
        if (ttgirFuncMatch) {
            logger.log(`[Trident] Extracted kernel name from tt.func: ${ttgirFuncMatch[1]}`);
            return ttgirFuncMatch[1];
        }

        // Try TTIR patterns (tt.func without @)
        const ttirFuncMatch = content.match(/tt\.func\s+(\w+)/);
        if (ttirFuncMatch) {
            logger.log(`[Trident] Extracted kernel name from tt.func (TTIR): ${ttirFuncMatch[1]}`);
            return ttirFuncMatch[1];
        }

        // Try filename as fallback (many files are named after the kernel)
        // BUT: Skip if it looks like a Python file or source file name
        const fileName = path.basename(filePath, path.extname(filePath));
        const ext = path.extname(filePath).toLowerCase();
        // Skip Python files and other source-like extensions
        if (ext === '.py' || ext === '.pyi' || ext === '.c' || ext === '.cpp' || ext === '.h' || ext === '.hpp') {
            // Don't use source file names as kernel names
            logger.log(`[Trident] Skipping filename fallback for source file: ${fileName}${ext}`);
        } else if (fileName && fileName.length < 50 && /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(fileName)) {
            // Only use filename if it looks like a valid identifier and is NOT a source file
            logger.log(`[Trident] Extracted kernel name from filename: ${fileName}`);
            return fileName;
        }

        // Fall back to directory name
        const dirName = path.basename(path.dirname(filePath));
        logger.log(`[Trident] Falling back to directory name as kernel name: ${dirName}`);
        return dirName;
    }

    /**
     * Parse callsite annotations from TTIR/TTGIR content
     * Format: #locN = loc(callsite(#locA at #locB))
     * Where #locA is the inlined location and #locB is the callsite
     * Handles nested callsites recursively
     */
    public parseCallsites(content: string): CallsiteInfo {
        const callsiteMap = new Map<string, string[]>();

        // First, parse all location definitions
        // Format: #loc50 = loc("/path/to/file.py":1500:24)
        const locPattern = /#loc(\d+)\s*=\s*loc\("([^"]+)":(\d+):\d+\)/g;
        const locMap = new Map<string, { file: string; line: number }>();

        let match;
        while ((match = locPattern.exec(content)) !== null) {
            const locId = `#loc${match[1]}`;
            const filePath = match[2];
            const lineNum = parseInt(match[3]);
            locMap.set(locId, { file: path.normalize(filePath), line: lineNum });
        }

        // Parse callsite annotations and build a callsite graph
        // Format: #loc75 = loc(callsite(#loc60 at #loc52))
        const callsitePattern = /#loc(\d+)\s*=\s*loc\(callsite\((#loc\d+)\s+at\s+(#loc\d+)\)\)/g;
        const callsiteGraph = new Map<string, { inlinedLocId: string; callsiteLocId: string }>();

        while ((match = callsitePattern.exec(content)) !== null) {
            const resultLocId = `#loc${match[1]}`;  // #loc31 (the resulting location)
            const inlinedLocId = match[2];          // #loc23 (the inlined code location - may itself be a callsite)
            const callsiteLocId = match[3];         // #loc12 (where the call happened)

            callsiteGraph.set(resultLocId, { inlinedLocId, callsiteLocId });
        }

        // Recursively resolve callsites to get all leaf locations
        const resolveCallsite = (locId: string, visited: Set<string> = new Set()): string[] => {
            if (visited.has(locId)) {
                return []; // Prevent infinite loops
            }
            visited.add(locId);

            // Check if this is a regular location
            const directLoc = locMap.get(locId);
            if (directLoc) {
                const key = `${directLoc.file}:${directLoc.line - 1}`;
                return [key];
            }

            // Check if this is a callsite that needs resolution
            const callsite = callsiteGraph.get(locId);
            if (callsite) {
                // Recursively resolve the inlined location
                return resolveCallsite(callsite.inlinedLocId, visited);
            }

            return [];
        };

        // Build the final callsite map by resolving all callsites
        for (const [_resultLocId, { inlinedLocId, callsiteLocId }] of callsiteGraph.entries()) {
            const callsiteLoc = locMap.get(callsiteLocId);

            if (callsiteLoc) {
                const callsiteKey = `${callsiteLoc.file}:${callsiteLoc.line - 1}`;

                if (!callsiteMap.has(callsiteKey)) {
                    callsiteMap.set(callsiteKey, []);
                }

                // Recursively resolve the inlined location
                const resolvedLocations = resolveCallsite(inlinedLocId);

                // Add all resolved locations
                const existing = callsiteMap.get(callsiteKey)!;
                for (const loc of resolvedLocations) {
                    if (!existing.includes(loc)) {
                        existing.push(loc);
                    }
                }
            }
        }

        // Log results
        if (callsiteMap.size > 0) {
            logger.log(`[Trident] Found ${callsiteMap.size} callsites with inlined code (nested resolved)`);
            for (const [callsite, inlined] of callsiteMap.entries()) {
                logger.log(`  ${path.basename(callsite)} → ${inlined.length} inlined locations`);
            }
        }

        return { callsiteMap };
    }

}

