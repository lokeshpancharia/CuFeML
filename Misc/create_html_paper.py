#!/usr/bin/env python3
"""
Convert Markdown research paper to HTML with proper academic formatting
"""

import markdown
from pathlib import Path
import re

def convert_markdown_to_html(markdown_file, output_html):
    """
    Convert markdown file to HTML with academic paper formatting
    """
    
    # Read the markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    html_content = md.convert(markdown_content)
    
    # Create a complete HTML document with CSS styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Machine Learning-Based Prediction of Compressive Strength in Cu-Fe High Entropy Alloys</title>
        <style>
            @media print {{
                @page {{
                    size: A4;
                    margin: 2.5cm 2cm 2.5cm 2cm;
                }}
                
                body {{
                    font-size: 11pt;
                    line-height: 1.4;
                }}
                
                h1 {{
                    font-size: 16pt;
                    page-break-after: avoid;
                }}
                
                h2 {{
                    font-size: 14pt;
                    page-break-after: avoid;
                    margin-top: 1.5em;
                }}
                
                h3 {{
                    font-size: 12pt;
                    page-break-after: avoid;
                }}
                
                img {{
                    max-width: 100%;
                    page-break-inside: avoid;
                }}
                
                table {{
                    page-break-inside: avoid;
                }}
                
                .page-break {{
                    page-break-before: always;
                }}
            }}
            
            body {{
                font-family: 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 2em;
                background-color: white;
            }}
            
            h1 {{
                font-size: 18pt;
                font-weight: bold;
                text-align: center;
                margin: 0 0 1.5em 0;
                color: #2c3e50;
            }}
            
            h2 {{
                font-size: 16pt;
                font-weight: bold;
                margin: 2em 0 1em 0;
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.3em;
            }}
            
            h3 {{
                font-size: 14pt;
                font-weight: bold;
                margin: 1.5em 0 0.8em 0;
                color: #2c3e50;
            }}
            
            h4 {{
                font-size: 12pt;
                font-weight: bold;
                margin: 1.2em 0 0.6em 0;
                color: #2c3e50;
            }}
            
            p {{
                margin: 0 0 1em 0;
                text-align: justify;
            }}
            
            .abstract {{
                margin: 2em 0;
                padding: 1.5em;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 5px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1.5em 0;
                font-size: 11pt;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1.5em auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .figure-caption {{
                font-size: 11pt;
                font-style: italic;
                text-align: center;
                margin: 0.5em 0 2em 0;
                color: #666;
                font-weight: 500;
            }}
            
            ol, ul {{
                margin: 1em 0;
                padding-left: 2em;
            }}
            
            li {{
                margin: 0.5em 0;
            }}
            
            code {{
                font-family: 'Courier New', monospace;
                background-color: #f1f2f6;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10pt;
            }}
            
            pre {{
                background-color: #f1f2f6;
                padding: 1.5em;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 10pt;
                border-left: 4px solid #3498db;
            }}
            
            .keywords {{
                font-weight: bold;
                margin-top: 1em;
                color: #2c3e50;
            }}
            
            .references {{
                font-size: 11pt;
            }}
            
            .references ol {{
                padding-left: 1.5em;
            }}
            
            .references li {{
                margin: 0.8em 0;
                text-align: justify;
            }}
            
            .toc {{
                background-color: #f8f9fa;
                padding: 1.5em;
                border-radius: 5px;
                margin: 2em 0;
                border: 1px solid #e9ecef;
            }}
            
            .toc h2 {{
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 1px solid #3498db;
            }}
            
            .highlight {{
                background-color: #fff3cd;
                padding: 0.2em 0.4em;
                border-radius: 3px;
            }}
            
            .note {{
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 1em;
                margin: 1em 0;
            }}
            
            .note::before {{
                content: "📝 Note: ";
                font-weight: bold;
                color: #0c5460;
            }}
        </style>
    </head>
    <body>
        {html_content}
        
        <div style="margin-top: 3em; padding-top: 2em; border-top: 2px solid #3498db; text-align: center; color: #666; font-size: 10pt;">
            <p>Generated from Markdown • {Path(markdown_file).name} • Machine Learning Research Paper</p>
        </div>
    </body>
    </html>
    """
    
    # Process the HTML to add figure captions and improve formatting
    html_template = process_figures_html(html_template)
    
    # Write HTML file
    try:
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Successfully converted {markdown_file} to {output_html}")
        return True
    except Exception as e:
        print(f"Error creating HTML: {e}")
        return False

def process_figures_html(html_content):
    """
    Process figure references and captions in the HTML content
    """
    # Replace figure markdown with proper HTML structure
    # Pattern 1: img followed by italic caption
    figure_pattern1 = r'<p><img src="([^"]+)" alt="([^"]*)" /></p>\s*<p><em>([^<]+)</em></p>'
    
    def replace_figure1(match):
        image_path = match.group(1)
        alt_text = match.group(2)
        caption = match.group(3)
        
        return f'''
        <div class="figure">
            <img src="{image_path}" alt="{alt_text}" />
            <p class="figure-caption">{caption}</p>
        </div>
        '''
    
    html_content = re.sub(figure_pattern1, replace_figure1, html_content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 2: img and caption in same paragraph
    figure_pattern2 = r'<p><img alt="([^"]*)" src="([^"]+)" />\s*<em>([^<]+)</em></p>'
    
    def replace_figure2(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        caption = match.group(3)
        
        return f'''
        <div class="figure">
            <img src="{image_path}" alt="{alt_text}" />
            <p class="figure-caption">{caption}</p>
        </div>
        '''
    
    html_content = re.sub(figure_pattern2, replace_figure2, html_content, flags=re.MULTILINE | re.DOTALL)
    
    # Add abstract styling
    html_content = html_content.replace('<h2>Abstract</h2>', '<h2>Abstract</h2><div class="abstract">')
    
    # Find the end of abstract (before Introduction)
    abstract_end = html_content.find('<h2>1. Introduction</h2>')
    if abstract_end != -1:
        # Insert closing div before Introduction
        html_content = html_content[:abstract_end] + '</div>\n\n' + html_content[abstract_end:]
    
    # Style keywords
    html_content = html_content.replace('<p><strong>Keywords:</strong>', '<p class="keywords"><strong>Keywords:</strong>')
    
    # Style references section
    html_content = html_content.replace('<h2>References</h2>', '<div class="references"><h2>References</h2>')
    html_content = html_content.replace('<h2>List of Figures</h2>', '</div><h2>List of Figures</h2>')
    
    return html_content

def main():
    """Main function to convert the research paper"""
    markdown_file = "research_paper_draft.md"
    output_html = "research_paper_draft.html"
    
    if not Path(markdown_file).exists():
        print(f"Error: {markdown_file} not found!")
        return
    
    print(f"Converting {markdown_file} to HTML...")
    success = convert_markdown_to_html(markdown_file, output_html)
    
    if success:
        print(f"HTML created successfully: {output_html}")
        print(f"File size: {Path(output_html).stat().st_size / 1024:.1f} KB")
        print("\nTo create PDF:")
        print("1. Open the HTML file in your browser")
        print("2. Press Ctrl+P (or Cmd+P on Mac)")
        print("3. Select 'Save as PDF' as destination")
        print("4. Choose appropriate settings and save")
        print("\nThe HTML file is optimized for printing with proper page breaks and formatting.")
    else:
        print("Failed to create HTML")

if __name__ == "__main__":
    main()