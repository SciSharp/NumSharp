#!/bin/bash
# Auto-generate llms.txt and llms-full.txt from actual documentation
# Parses toc.yml, markdown files, and API metadata - no hardcoded content

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$SRC_DIR/../website}"
SITE_URL="${2:-https://scisharp.github.io/NumSharp}"

echo "Generating AI-friendly documentation files..."
echo "Source: $SRC_DIR"
echo "Output: $OUTPUT_DIR"

# ============================================================================
# Helper: Extract first paragraph from markdown (description)
# ============================================================================
extract_description() {
    local file="$1"
    # Skip YAML frontmatter, get first non-empty paragraph
    awk '
        BEGIN { in_frontmatter=0; found=0 }
        /^---$/ {
            if (NR==1) { in_frontmatter=1; next }
            else { in_frontmatter=0; next }
        }
        in_frontmatter { next }
        /^[ \t]*#/ { next }  # Skip headers (with optional leading whitespace)
        /^[ \t]*$/ { if (found) exit; next }  # Empty line after content = done
        /^[^-\*\|>]/ {
            gsub(/\*\*/, ""); gsub(/\*/, ""); gsub(/`/, "")  # Remove markdown formatting
            gsub(/^[ \t]+/, "")  # Trim leading whitespace
            print; found=1
        }
    ' "$file" | head -2 | tr '\n' ' ' | sed 's/  */ /g' | head -c 200
}

# ============================================================================
# Helper: Extract title from markdown (first H1)
# ============================================================================
extract_title() {
    local file="$1"
    # Handle optional leading whitespace before #
    grep -m1 "^[ \t]*# " "$file" 2>/dev/null | sed 's/^[ \t]*# //' | sed 's/[ \t]*$//' || basename "$file" .md
}

# ============================================================================
# Helper: Parse toc.yml and generate links
# ============================================================================
parse_toc() {
    local toc_file="$1"
    local base_url="$2"
    local base_path="$3"

    if [ ! -f "$toc_file" ]; then
        return
    fi

    # Simple YAML parser for toc.yml format
    awk -v base_url="$base_url" -v base_path="$base_path" '
        /^- name:/ {
            name = $0
            gsub(/^- name: */, "", name)
            gsub(/["'\'']/, "", name)
        }
        /^  href:/ {
            href = $0
            gsub(/^  href: */, "", href)
            gsub(/["'\'']/, "", href)
            # Convert .md to .html
            gsub(/\.md$/, ".html", href)
            if (href !~ /^http/) {
                href = base_url "/" base_path "/" href
            }
            print "- [" name "](" href ")"
        }
    ' "$toc_file"
}

# ============================================================================
# Generate llms.txt
# ============================================================================
generate_llms_txt() {
    local output="$OUTPUT_DIR/llms.txt"

    # Header from index.md or default
    echo "# NumSharp" > "$output"
    echo "" >> "$output"

    # Extract description from index.md
    if [ -f "$SRC_DIR/index.md" ]; then
        echo "> $(extract_description "$SRC_DIR/index.md")" >> "$output"
    else
        echo "> NumSharp is a .NET port of Python's NumPy library for numerical computing." >> "$output"
    fi
    echo "" >> "$output"

    # Quick start section
    cat >> "$output" << 'QUICKSTART'
## Installation

```bash
dotnet add package NumSharp
```

## Quick Start

```csharp
using NumSharp;

var a = np.array(new[] { 1, 2, 3, 4, 5 });
var b = np.zeros((3, 4));
var result = np.sum(a);
var slice = a["1:4"];  // Slicing returns views
```

QUICKSTART

    # Documentation section - parse docs/toc.yml
    echo "## Documentation" >> "$output"
    echo "" >> "$output"

    if [ -d "$SRC_DIR/docs" ]; then
        for md_file in "$SRC_DIR/docs/"*.md; do
            if [ -f "$md_file" ]; then
                filename=$(basename "$md_file" .md)
                title=$(extract_title "$md_file")
                desc=$(extract_description "$md_file")
                echo "- [$title](${SITE_URL}/docs/${filename}.html): $desc" >> "$output"
            fi
        done
    elif [ -d "$SRC_DIR/articles" ]; then
        for md_file in "$SRC_DIR/articles/"*.md; do
            if [ -f "$md_file" ]; then
                filename=$(basename "$md_file" .md)
                title=$(extract_title "$md_file")
                desc=$(extract_description "$md_file")
                echo "- [$title](${SITE_URL}/articles/${filename}.html): $desc" >> "$output"
            fi
        done
    fi
    echo "" >> "$output"

    # API Reference section - parse generated API yml files
    echo "## API Reference" >> "$output"
    echo "" >> "$output"

    # Find key API types from generated yml or fallback
    if [ -d "$SRC_DIR/api" ]; then
        # Look for key types in yml files
        for yml_file in "$SRC_DIR/api/"*.yml; do
            if [ -f "$yml_file" ]; then
                # Extract uid and summary from yml
                uid=$(grep -m1 "^uid:" "$yml_file" 2>/dev/null | sed 's/uid: *//' || true)
                summary=$(grep -m1 "summary:" "$yml_file" 2>/dev/null | sed 's/summary: *//' | sed 's/^"//' | sed 's/"$//' | head -c 100 || true)

                if [ -n "$uid" ] && [[ "$uid" == NumSharp* ]]; then
                    # Only include top-level types, not nested
                    if [[ "$uid" =~ ^NumSharp\.[A-Z][a-zA-Z0-9]*$ ]] || [[ "$uid" == "NumSharp" ]]; then
                        html_name=$(echo "$uid" | sed 's/\./\./g')
                        if [ -n "$summary" ]; then
                            echo "- [@$uid](${SITE_URL}/api/${uid}.html): $summary" >> "$output"
                        else
                            echo "- [@$uid](${SITE_URL}/api/${uid}.html)" >> "$output"
                        fi
                    fi
                fi
            fi
        done 2>/dev/null | head -20
    fi

    # If no API yml found, add placeholder links
    if ! grep -q "api/NumSharp" "$output" 2>/dev/null; then
        cat >> "$output" << 'API_FALLBACK'
- [NumSharp.NDArray](https://scisharp.github.io/NumSharp/api/NumSharp.NDArray.html): Main n-dimensional array type
- [NumSharp.np](https://scisharp.github.io/NumSharp/api/NumSharp.np.html): Static NumPy-style API
- [NumSharp.Shape](https://scisharp.github.io/NumSharp/api/NumSharp.Shape.html): Array dimensions and strides
API_FALLBACK
    fi
    echo "" >> "$output"

    # Data types section - extract from codebase or use known types
    cat >> "$output" << 'TYPES'
## Supported Data Types

bool, byte, short, ushort, int, uint, long, ulong, float, double, decimal, char

## Key Concepts

- **NDArray**: Multi-dimensional array in unmanaged memory
- **Shape**: Dimensions and strides for offset calculation
- **Broadcasting**: Arrays with different shapes operate element-wise
- **Views**: Slicing returns views (shared memory), use `.copy()` for copies

TYPES

    # Optional section
    echo "## Optional" >> "$output"
    echo "" >> "$output"
    echo "- [Full API Reference](${SITE_URL}/api/): Complete class and method documentation" >> "$output"
    echo "- [GitHub Repository](https://github.com/SciSharp/NumSharp): Source code and issues" >> "$output"
    echo "- [NuGet Package](https://www.nuget.org/packages/NumSharp): Latest releases" >> "$output"

    echo "Generated: $output ($(wc -l < "$output") lines)"
}

# ============================================================================
# Generate llms-full.txt
# ============================================================================
generate_llms_full_txt() {
    local output="$OUTPUT_DIR/llms-full.txt"

    echo "# NumSharp - Complete Documentation" > "$output"
    echo "" >> "$output"
    echo "> This file contains the complete NumSharp documentation for AI/LLM ingestion." >> "$output"
    echo "> Auto-generated from source markdown files." >> "$output"
    echo "" >> "$output"
    echo "---" >> "$output"
    echo "" >> "$output"

    # Table of contents
    echo "## Table of Contents" >> "$output"
    echo "" >> "$output"

    local toc_num=1
    local docs_dir="$SRC_DIR/docs"
    [ ! -d "$docs_dir" ] && docs_dir="$SRC_DIR/articles"

    if [ -d "$docs_dir" ]; then
        for md_file in "$docs_dir/"*.md; do
            if [ -f "$md_file" ] && [[ "$(basename "$md_file")" != "toc.yml" ]]; then
                title=$(extract_title "$md_file")
                echo "$toc_num. $title" >> "$output"
                ((toc_num++))
            fi
        done
    fi
    echo "" >> "$output"
    echo "---" >> "$output"

    # Concatenate all documentation files
    if [ -d "$docs_dir" ]; then
        for md_file in "$docs_dir/"*.md; do
            if [ -f "$md_file" ] && [[ "$(basename "$md_file")" != "toc.yml" ]]; then
                echo "" >> "$output"
                # Strip YAML frontmatter and include content
                awk '
                    BEGIN { in_frontmatter=0; skip_first=1 }
                    /^---$/ {
                        if (NR==1) { in_frontmatter=1; next }
                        else { in_frontmatter=0; next }
                    }
                    in_frontmatter { next }
                    { print }
                ' "$md_file" >> "$output"
                echo "" >> "$output"
                echo "---" >> "$output"
            fi
        done
    fi

    # Add API summary from yml files if available
    echo "" >> "$output"
    echo "# API Quick Reference" >> "$output"
    echo "" >> "$output"

    if [ -d "$SRC_DIR/api" ] && ls "$SRC_DIR/api/"*.yml >/dev/null 2>&1; then
        echo "## Namespaces and Types" >> "$output"
        echo "" >> "$output"

        # Parse yml files for type information
        for yml_file in "$SRC_DIR/api/"*.yml; do
            if [ -f "$yml_file" ]; then
                # Extract namespace/class info
                awk '
                    /^uid:/ { uid = $2 }
                    /^type:/ { type = $2 }
                    /^summary:/ {
                        summary = $0
                        gsub(/^summary: *["'\'']?/, "", summary)
                        gsub(/["'\'']$/, "", summary)
                    }
                    END {
                        if (uid && type) {
                            if (type == "Namespace" || type == "Class" || type == "Struct") {
                                printf "- **%s** (%s): %s\n", uid, type, summary
                            }
                        }
                    }
                ' "$yml_file" 2>/dev/null
            fi
        done | sort | uniq | head -50 >> "$output"
    else
        # Fallback: Include common API patterns
        cat >> "$output" << 'API_PATTERNS'
## Common API Patterns

### Array Creation (np.*)
- `np.array(data)` - Create from existing data
- `np.zeros(shape)`, `np.ones(shape)`, `np.empty(shape)` - Initialize arrays
- `np.arange(start, stop, step)` - Range of values
- `np.linspace(start, stop, num)` - Evenly spaced values
- `np.eye(n)` - Identity matrix
- `np.random.rand(shape)`, `np.random.randn(shape)` - Random arrays

### Math Operations
- `np.sum(a)`, `np.mean(a)`, `np.std(a)` - Statistics
- `np.dot(a, b)`, `np.matmul(a, b)` - Linear algebra
- `np.sqrt(a)`, `np.exp(a)`, `np.log(a)` - Element-wise math
- `+`, `-`, `*`, `/`, `%` - Arithmetic operators (with broadcasting)

### Array Manipulation
- `np.reshape(a, shape)`, `a.reshape(shape)` - Change shape
- `np.transpose(a)`, `a.T` - Transpose
- `np.concatenate([a, b])`, `np.stack([a, b])` - Join arrays
- `np.squeeze(a)`, `np.expand_dims(a, axis)` - Dimension changes

### Indexing
- `a[i]`, `a[i, j]` - Element access
- `a["start:stop"]`, `a["::step"]` - Slicing (returns view)
- `a[boolArray]` - Boolean mask
- `a[intArray]` - Fancy indexing
API_PATTERNS
    fi

    echo "" >> "$output"
    echo "---" >> "$output"
    echo "" >> "$output"
    echo "*Auto-generated from NumSharp documentation source files*" >> "$output"

    echo "Generated: $output ($(wc -l < "$output") lines)"
}

# ============================================================================
# Generate robots.txt
# ============================================================================
generate_robots_txt() {
    local output="$OUTPUT_DIR/robots.txt"

    cat > "$output" << ROBOTS
# NumSharp Documentation - robots.txt
# Auto-generated - Allow all crawlers including AI

User-agent: *
Allow: /

# AI Crawlers - explicitly allowed
User-agent: GPTBot
Allow: /

User-agent: ChatGPT-User
Allow: /

User-agent: Claude-Web
Allow: /

User-agent: ClaudeBot
Allow: /

User-agent: Anthropic-AI
Allow: /

User-agent: Google-Extended
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Amazonbot
Allow: /

User-agent: Bytespider
Allow: /

User-agent: CCBot
Allow: /

# AI-friendly documentation files
# ${SITE_URL}/llms.txt - Curated summary
# ${SITE_URL}/llms-full.txt - Complete documentation

Sitemap: ${SITE_URL}/sitemap.xml
ROBOTS

    echo "Generated: $output"
}

# ============================================================================
# Main
# ============================================================================

mkdir -p "$OUTPUT_DIR"

generate_llms_txt
generate_llms_full_txt
generate_robots_txt

echo ""
echo "AI-friendly documentation generation complete!"
echo "Files created in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR/llms.txt" "$OUTPUT_DIR/llms-full.txt" "$OUTPUT_DIR/robots.txt" 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}'
