import * as vscode from 'vscode';

class Logger {
    private outputChannel: vscode.OutputChannel;
    private extensionName = 'Trident';

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel(this.extensionName);
    }

    public log(message: string) {
        const timestamp = new Date().toLocaleTimeString();
        this.outputChannel.appendLine(`[${timestamp}] ${this.extensionName}: ${message}`);
    }

    public error(message: string, error?: Error | unknown) {
        const timestamp = new Date().toLocaleTimeString();
        this.outputChannel.appendLine(`[${timestamp}] ${this.extensionName} ERROR: ${message}`);
        if (error) {
            this.outputChannel.appendLine(`  ${error}`);
            if (error instanceof Error && error.stack) {
                this.outputChannel.appendLine(`  Stack: ${error.stack}`);
            }
        }
    }

    public show() {
        this.outputChannel.show();
    }

    public dispose() {
        this.outputChannel.dispose();
    }
}

export const logger = new Logger();

