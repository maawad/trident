# Trident 🔱

<p align="center">
  <img src="resources/icon.png" alt="Trident Icon" width="128" height="128">
</p>

**Navigate Triton GPU assembly with bidirectional source-assembly highlighting and powerful inspection tools.**

Trident is a VS Code extension that provides seamless visualization and navigation of AMD GCN assembly output from Triton GPU kernels. With bidirectional highlighting, powerful filtering, and intelligent diff viewing, Trident makes understanding low-level GPU code effortless.

## ✨ Features

### 🔄 Bidirectional Highlighting
- Click on Python/Triton source code to highlight corresponding assembly instructions
- Click on assembly instructions to jump to the source code
- Navigate between multiple assembly blocks mapped to a single source line
- Automatic focus management keeps your workflow smooth

### 📊 Assembly Viewer
- Side-by-side view of source and assembly
- Syntax highlighting for AMD GCN assembly
- Theme-aware styling that adapts to your VS Code theme
- Line numbers for easy reference
- Jump to top/bottom navigation

### 🔍 Advanced Filtering
- **Godbolt-style filters**: Hide directives, comments, and empty lines
- **Dynamic instruction filtering**: Automatically extracts and filters specific GCN instructions
- **Searchable filters**: Find instructions quickly with built-in search
- Select all/none for quick filter management
- Smart renumbering when filters are active

### 🔎 In-Webview Search
- Press `Ctrl+F` to search within assembly
- Navigate through matches with up/down buttons
- Highlight all matches with subtle, theme-aware colors
- Works in both normal and diff views

### 📈 Kernel Version Management
- View all cached versions of a kernel with timestamps
- Compare different kernel versions side-by-side
- Custom diff viewer preserves all interactive features
- Track optimization changes over time

### 🎯 Smart Cache Integration
- Automatically reads from `~/.triton/cache`
- Configurable cache directory
- Shows cache file paths with timestamps
- Click cache path to open the assembly file directly

### 🚀 Workspace Integration
- Context menu integration: Right-click in Python files to show assembly
- Editor toolbar button for quick access
- Tracks all open Python files with Triton kernels
- Persistent state across VS Code reloads

## 📦 Installation

### From VSIX (Local)
1. Download the `.vsix` file
2. In VS Code: `Extensions` → `...` → `Install from VSIX`
3. Select the downloaded file

### From Marketplace (Coming Soon)
Search for "Trident" in the VS Code Extensions marketplace

## 🚀 Quick Start

1. Open a Python file with Triton kernels (decorated with `@triton.jit`)
2. Right-click in the editor and select `Trident: Show GCN Assembly`
   - Or click the chip icon (🔲) in the editor toolbar
3. The assembly viewer will open side-by-side
4. Click on any source line to see the corresponding assembly
5. Click on assembly instructions to jump back to source

## ⚙️ Configuration

### Cache Directory
By default, Trident reads from `~/.triton/cache`. You can customize this:

```json
{
  "trident.cacheDirectory": "~/custom/triton/cache"
}
```

Supports `~` expansion for home directory.

## 🎮 Usage Tips

### Navigation
- Use the kernel dropdown to switch between different kernels
- Use the file dropdown to view assembly from different source files
- Use chevron buttons (↑↓) to navigate between assembly blocks for a single source line

### Filtering
- Click the filter icon to open the filter menu
- Enable/disable directives, comments, and empty lines
- Search and toggle specific instructions
- Filters persist when switching between views

### Comparing Versions
- Click the diff icon next to any kernel in the dropdown
- Select another version to compare
- All highlighting and filtering features work in diff view
- Use search to find specific changes

### Search
- Press `Ctrl+F` in the assembly viewer
- Type your search term
- Use ↑/↓ buttons or `Enter`/`Shift+Enter` to navigate
- Search works with active filters

## 🏗️ Architecture

Trident consists of several key components:

- **Assembly Viewer**: WebView panel with interactive assembly display
- **Triton Compiler Integration**: Reads and parses GCN assembly from cache
- **Highlighting Manager**: Manages bidirectional source-assembly highlighting
- **Line Mapping**: Uses `.loc` and `.file` directives for accurate mapping

## 🔧 Development

### Prerequisites
- Node.js 20.x or higher
- VS Code 1.85.0 or higher

### Building
```bash
npm install
npm run compile
```

### Debugging
Press `F5` in VS Code to launch the extension in debug mode.

### Testing
```bash
npm test
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [Godbolt Compiler Explorer](https://godbolt.org/)
- Built for the [Triton](https://github.com/openai/triton) GPU programming language
- Designed for AMD GCN assembly analysis

## 📬 Contact

Muhammad Awad - [@maawad](https://github.com/maawad)

Project Link: [https://github.com/maawad/Trident](https://github.com/maawad/Trident)

---

**Made with 🔱 for GPU kernel developers**
