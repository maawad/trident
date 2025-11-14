import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { promisify } from 'util';

const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const stat = promisify(fs.stat);
const unlink = promisify(fs.unlink);

export interface KernelAssembly {
    kernelName: string;
    assembly: string;
    cachePath: string;
    timestamp: Date;
    sourceFiles: string[];  // All source files from .file directives
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
     * Extract all source file paths from .file directives in assembly
     */
    extractSourceFiles(assembly: string): string[] {
        const sourceFiles: string[] = [];

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

        return sourceFiles;
    }

    /**
     * Extract kernel names from Python source code
     */
    extractKernelNames(pythonSource: string): string[] {
        const kernelNames: string[] = [];

        // Match @triton.jit decorated functions
        const regex = /@triton\.jit\s+def\s+(\w+)/g;
        let match;

        while ((match = regex.exec(pythonSource)) !== null) {
            kernelNames.push(match[1]);
        }

        return kernelNames;
    }

    /**
     * Search the Triton cache for AMD GCN assembly files matching the source file
     */
    async findCachedAssemblies(sourceFilePath?: string, workspaceFolders?: string[]): Promise<KernelAssembly[]> {
        // Update cache path from configuration in case it changed
        this.updateCachePath();

        let assemblies: KernelAssembly[] = [];

        try {
            if (!fs.existsSync(this.tritonCachePath)) {
                throw new Error(`Triton cache directory not found: ${this.tritonCachePath}`);
            }

            const entries = await readdir(this.tritonCachePath);

            for (const entry of entries) {
                const entryPath = path.join(this.tritonCachePath, entry);
                const entryStat = await stat(entryPath);

                if (entryStat.isDirectory()) {
                    // Search for GCN assembly files in this cache directory
                    const gcnFiles = await this.findGCNFiles(entryPath);

                    for (const gcnFile of gcnFiles) {
                        const content = await readFile(gcnFile, 'utf-8');
                        const kernelName = this.extractKernelName(gcnFile, content);
                        const sourceFiles = this.extractSourceFiles(content);

                        assemblies.push({
                            kernelName,
                            assembly: content,
                            cachePath: gcnFile,
                            timestamp: entryStat.mtime,
                            sourceFiles
                        });
                    }
                }
            }

            // Sort by most recent first
            assemblies.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

            // Filter to only include assemblies from workspace files
            if (workspaceFolders && workspaceFolders.length > 0) {
                assemblies = assemblies.filter(asm => {
                    // Check if ANY source file is in the workspace
                    return asm.sourceFiles.some(srcFile => {
                        return workspaceFolders.some(wsFolder =>
                            srcFile.startsWith(wsFolder)
                        );
                    });
                });
            }

            // Filter by source file - show all kernels from the source file
            if (sourceFilePath) {
                const filtered = assemblies.filter(asm => {
                    if (!asm.sourceFiles || asm.sourceFiles.length === 0) {
                        return false;
                    }
                    // Check if any of the source files match
                    return asm.sourceFiles.some(srcFile => {
                        // Match by exact path or just filename
                        return srcFile === sourceFilePath ||
                            srcFile.endsWith('/' + path.basename(sourceFilePath)) ||
                            path.basename(srcFile) === path.basename(sourceFilePath);
                    });
                });

                // Only apply filter if we found matches, otherwise show all
                if (filtered.length > 0) {
                    assemblies = filtered;
                }
            }

            return assemblies;

        } catch (error) {
            console.error('Error searching Triton cache:', error);
            throw error;
        }

        return assemblies;
    }

    /**
     * Find all GCN assembly files in a directory
     */
    private async findGCNFiles(dir: string): Promise<string[]> {
        const gcnFiles: string[] = [];

        try {
            const files = await readdir(dir);

            for (const file of files) {
                const filePath = path.join(dir, file);
                const fileStat = await stat(filePath);

                if (fileStat.isFile()) {
                    // Look for files that contain GCN assembly
                    // Common extensions: .s, .asm, .amdgcn, or files without extension
                    if (file.endsWith('.s') ||
                        file.endsWith('.asm') ||
                        file.endsWith('.amdgcn') ||
                        file.includes('amdgcn') ||
                        file.includes('gcn')) {
                        gcnFiles.push(filePath);
                    } else if (!file.includes('.')) {
                        // Check content for GCN assembly markers
                        const content = await readFile(filePath, 'utf-8');
                        if (this.isGCNAssembly(content)) {
                            gcnFiles.push(filePath);
                        }
                    }
                }
            }
        } catch (error) {
            console.error(`Error reading directory ${dir}:`, error);
        }

        return gcnFiles;
    }

    /**
     * Check if content looks like GCN assembly
     */
    private isGCNAssembly(content: string): boolean {
        const gcnMarkers = [
            '.amdgcn',
            '.amdgpu',
            's_load_',
            'v_mov_',
            'v_add_',
            's_waitcnt',
            '.hsa_code_object',
            'amdgpu_hsa_kernel',
            'gfx',
            '.amdhsa_'
        ];

        return gcnMarkers.some(marker => content.includes(marker));
    }

    /**
     * Extract kernel name from file path or content
     */
    private extractKernelName(filePath: string, content: string): string {
        // Try to extract from .amdhsa_kernel directive
        const kernelMatch = content.match(/\.amdhsa_kernel\s+(\w+)/);
        if (kernelMatch) {
            return kernelMatch[1];
        }

        // Try to extract from .globl directive
        const globalMatch = content.match(/\.globl\s+(\w+)/);
        if (globalMatch) {
            return globalMatch[1];
        }

        // Fall back to directory name or file name
        const dirName = path.basename(path.dirname(filePath));
        return dirName;
    }

}

