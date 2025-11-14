import * as path from 'path';
import { KernelAssembly } from './tritonCompiler';
import { logger } from './logger';

/**
 * Unified line mapping data structure
 */
export interface LineMapping {
    // Maps "filepath:line" -> array of assembly line numbers
    sourceToAsm: Map<string, number[]>;
    // Maps assembly line number -> source file path
    asmLineToFile: Map<number, string>;
}

/**
 * Modular line mapping builder
 * Preprocesses all mappings into a unified data structure
 */
export class LineMappingBuilder {
    private sourceToAsm: Map<string, number[]> = new Map();
    private asmLineToFile: Map<number, string> = new Map();
    private globalLineOffset = 0;

    /**
     * Build unified line mapping for all assemblies
     */
    public buildMapping(assemblies: KernelAssembly[]): LineMapping {
        this.sourceToAsm.clear();
        this.asmLineToFile.clear();
        this.globalLineOffset = 0;

        for (const asm of assemblies) {
            const lines = asm.assembly.split('\n');

            switch (asm.irType) {
                case 'gcn':
                    this.buildGCNMapping(asm, lines);
                    break;
                case 'ttgir':
                case 'ttir':
                    this.buildTTGIRMapping(asm, lines);
                    break;
                case 'llvm':
                    this.buildLLVMMapping(asm, lines);
                    break;
                default:
                    logger.log(`[Trident] No line mapping support for IR type: ${asm.irType}`);
            }

            this.globalLineOffset += lines.length;
        }

        logger.log(`[Trident] Built line mapping with ${this.sourceToAsm.size} source line entries across all files`);

        return {
            sourceToAsm: this.sourceToAsm,
            asmLineToFile: this.asmLineToFile
        };
    }

    private buildGCNMapping(asm: KernelAssembly, lines: string[]): void {
        let currentSourceLine = -1;
        let currentFileId = -1;

        // First pass: build file ID to path mapping
        const fileMap = new Map<number, string>();
        for (const line of lines) {
            // Format 2: .file <id> "dir" "file"
            const format2Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"\s+"([^"]+)"/);
            if (format2Match) {
                const fileId = parseInt(format2Match[1]);
                const directory = format2Match[2].replace(/^;/, '');
                const filename = format2Match[3];
                fileMap.set(fileId, path.join(directory, filename));
                continue;
            }

            // Format 1: .file <id> "path"
            const format1Match = line.match(/\.file\s+(\d+)\s+"([^"]+)"$/);
            if (format1Match) {
                const fileId = parseInt(format1Match[1]);
                fileMap.set(fileId, format1Match[2]);
            }
        }

        // Second pass: map source lines to assembly lines
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const globalLineNum = this.globalLineOffset + i;

            // Parse .loc directive: .loc <file_id> <line> <column>
            const locMatch = line.match(/\.loc\s+(\d+)\s+(\d+)\s+\d+/);
            if (locMatch) {
                currentFileId = parseInt(locMatch[1]);
                currentSourceLine = parseInt(locMatch[2]) - 1; // Convert to 0-indexed
            }

            // Track which file this assembly line belongs to
            if (currentFileId >= 0) {
                const sourceFilePath = fileMap.get(currentFileId);
                if (sourceFilePath) {
                    this.asmLineToFile.set(globalLineNum, path.normalize(sourceFilePath));
                }
            }

            // Map this assembly line to source lines
            if (currentSourceLine >= 0 && currentFileId >= 0) {
                const sourceFilePath = fileMap.get(currentFileId);
                if (sourceFilePath) {
                    const normalizedPath = path.normalize(sourceFilePath);
                    const key = `${normalizedPath}:${currentSourceLine}`;

                    if (!this.sourceToAsm.has(key)) {
                        this.sourceToAsm.set(key, []);
                    }

                    // Only add actual instructions
                    const trimmed = line.trim();
                    const isDirective = trimmed.startsWith('.');
                    const isComment = trimmed.startsWith('#') || trimmed.startsWith(';') || trimmed.startsWith('//');
                    const isEmpty = trimmed.length === 0;

                    if (!isEmpty && !isDirective && !isComment) {
                        this.sourceToAsm.get(key)!.push(globalLineNum);
                    }
                }
            }
        }
    }

    private buildTTGIRMapping(asm: KernelAssembly, lines: string[]): void {
        let currentSourceLine = -1;
        let currentSourceFile: string | undefined;

        const locPattern1 = /#loc\s*=\s*loc\("([^"]+)":(\d+):\d+\)/;
        const locPattern2 = /loc\("([^"]+)":(\d+):\d+\)/;
        const locDefPattern = /#loc(\d+)\s*=\s*loc\("([^"]+)":(\d+):\d+\)/;
        const locRefPattern = /loc\(#loc(\d+)\)/;

        // First pass: collect all location definitions
        const locDefMap = new Map<number, { file: string; line: number }>();
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const defMatch = line.match(locDefPattern);
            if (defMatch) {
                const locId = parseInt(defMatch[1]);
                const filePath = defMatch[2];
                const lineNum = parseInt(defMatch[3]) - 1;
                locDefMap.set(locId, { file: filePath, line: lineNum });
            }
        }

        // Second pass: process lines and resolve location references
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const globalLineNum = this.globalLineOffset + i;

            // Try pattern 1 first (#loc = loc(...))
            let locMatch = line.match(locPattern1);
            if (locMatch) {
                currentSourceFile = locMatch[1];
                currentSourceLine = parseInt(locMatch[2]) - 1;
            } else {
                // Try pattern 2 (just loc(...))
                locMatch = line.match(locPattern2);
                if (locMatch) {
                    currentSourceFile = locMatch[1];
                    currentSourceLine = parseInt(locMatch[2]) - 1;
                } else {
                    // Try pattern 3 (loc(#locN) reference)
                    const refMatch = line.match(locRefPattern);
                    if (refMatch) {
                        const locId = parseInt(refMatch[1]);
                        const locDef = locDefMap.get(locId);
                        if (locDef) {
                            currentSourceFile = locDef.file;
                            currentSourceLine = locDef.line;
                        }
                    }
                }
            }

            // Track which file this assembly line belongs to
            if (currentSourceFile) {
                this.asmLineToFile.set(globalLineNum, path.normalize(currentSourceFile));
            }

            // Map this assembly line to source lines
            if (currentSourceLine >= 0 && currentSourceFile) {
                const normalizedPath = path.normalize(currentSourceFile);
                const key = `${normalizedPath}:${currentSourceLine}`;

                if (!this.sourceToAsm.has(key)) {
                    this.sourceToAsm.set(key, []);
                }

                // Only add actual instructions
                const trimmed = line.trim();
                const isComment = trimmed.startsWith('#') && !trimmed.startsWith('#loc');
                const isEmpty = trimmed.length === 0;

                if (!isEmpty && !isComment) {
                    this.sourceToAsm.get(key)!.push(globalLineNum);
                }
            }
        }
    }

    private buildLLVMMapping(asm: KernelAssembly, lines: string[]): void {
        let currentSourceLine = -1;
        let currentSourceFile: string | undefined;

        const difilePattern = /!(\d+)\s*=\s*!DIFile\([^)]*filename:\s*"([^"]+)"[^)]*directory:\s*"([^"]+)"[^)]*\)/;
        const disubprogramPattern = /!(\d+)\s*=\s*(?:distinct\s+)?!DISubprogram\([^)]*\)/;
        const disubprogramFilePattern = /file:\s*!(\d+)/;
        const disubprogramLinePattern = /line:\s*(\d+)/;
        const dilocationPattern = /!(\d+)\s*=\s*!DILocation\([^)]*line:\s*(\d+)[^)]*scope:\s*!(\d+)[^)]*\)/;

        // First pass: collect all !DIFile metadata
        const fileMap = new Map<number, string>();
        for (const line of lines) {
            const fileMatch = line.match(difilePattern);
            if (fileMatch) {
                const fileId = parseInt(fileMatch[1]);
                const filename = fileMatch[2];
                const directory = fileMatch[3];
                const filePath = path.join(directory, filename);
                fileMap.set(fileId, filePath);
            }
        }

        // Second pass: collect !DISubprogram metadata
        const subprogramMap = new Map<number, { file: string; line: number }>();
        for (const line of lines) {
            const subprogramMatch = line.match(disubprogramPattern);
            if (subprogramMatch) {
                const subprogramId = parseInt(subprogramMatch[1]);
                const fileMatch = line.match(disubprogramFilePattern);
                const lineMatch = line.match(disubprogramLinePattern);
                if (fileMatch && lineMatch) {
                    const fileId = parseInt(fileMatch[1]);
                    const lineNum = parseInt(lineMatch[1]) - 1;
                    const filePath = fileMap.get(fileId);
                    if (filePath) {
                        subprogramMap.set(subprogramId, { file: filePath, line: lineNum });
                    }
                }
            }
        }

        // Third pass: collect ALL !DILocation definitions
        const locationMap = new Map<number, { line: number; scopeId: number }>();
        for (const line of lines) {
            const locMatch = line.match(dilocationPattern);
            if (locMatch) {
                const locationId = parseInt(locMatch[1]);
                const lineNum = parseInt(locMatch[2]) - 1;
                const scopeId = parseInt(locMatch[3]);
                locationMap.set(locationId, { line: lineNum, scopeId: scopeId });
            }
        }

        // Fourth pass: process lines and map !DILocation to source lines
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const globalLineNum = this.globalLineOffset + i;

            // Check if this line has a !DILocation reference
            const dbgMatch = line.match(/!dbg\s*!(\d+)/);
            if (dbgMatch) {
                const locationId = parseInt(dbgMatch[1]);
                const locationDef = locationMap.get(locationId);

                if (locationDef) {
                    const subprogram = subprogramMap.get(locationDef.scopeId);
                    if (subprogram) {
                        currentSourceFile = subprogram.file;
                        currentSourceLine = locationDef.line;
                    }
                }
            }

            // Track which file this assembly line belongs to
            if (currentSourceFile) {
                this.asmLineToFile.set(globalLineNum, path.normalize(currentSourceFile));
            }

            // Map this assembly line to source lines
            if (currentSourceLine >= 0 && currentSourceFile) {
                const normalizedPath = path.normalize(currentSourceFile);
                const key = `${normalizedPath}:${currentSourceLine}`;

                if (!this.sourceToAsm.has(key)) {
                    this.sourceToAsm.set(key, []);
                }

                // Only add actual instructions
                const trimmed = line.trim();
                const isComment = trimmed.startsWith(';') || (trimmed.startsWith('!') && !trimmed.startsWith('!dbg'));
                const isEmpty = trimmed.length === 0;

                if (!isEmpty && !isComment) {
                    this.sourceToAsm.get(key)!.push(globalLineNum);
                }
            }
        }
    }
}

