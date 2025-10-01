#!/usr/bin/env python3
"""
Convert Markdown research paper to PDF with proper formatting
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

def convert_markdown_to_pdf(markdown_file, output_pdf):
    """
    Convert markdown file to PDF with academic paper formatting
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
            @page {{
                size: A4;
                margin: 2.5cm 2cm 2.5cm 2cm;
                @bottom-center {{
                    content: counter(page);
                    font-size: 10pt;
                }}
            }}
            
            body {{
                font-family: 'Times New Roman', serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: none;
                margin: 0;
                padding: 0;
            }}
            
            h1 {{
                font-size: 16pt;
                font-weight: bold;
                text-align: center;
                margin: 0 0 1.5em 0;
                page-break-after: avoid;
            }}
            
            h2 {{
                font-size: 14pt;
                font-weight: bold;
                margin: 1.5em 0 0.8em 0;
                page-break-after: avoid;
            }}
            
            h3 {{
                font-size: 12pt;
                font-weight: bold;
                margin: 1.2em 0 0.6em 0;
                page-break-after: avoid;
            }}
            
            h4 {{
                font-size: 11pt;
                font-weight: bold;
                margin: 1em 0 0.5em 0;
                page-break-after: avoid;
            }}
            
            p {{
                margin: 0 0 0.8em 0;
                text-align: justify;
                orphans: 2;
                widows: 2;
            }}
            
            .abstract {{
                margin: 1.5em 0;
                padding: 1em;
                background-color: #f9f9f9;
                border-left: 4px solid #ccc;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1em 0;
                font-size: 10pt;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1em auto;
                page-break-inside: avoid;
            }}
            
            .figure-caption {{
                font-size: 10pt;
                font-style: italic;
                text-align: center;
                margin: 0.5em 0 1.5em 0;
                color: #666;
            }}
            
            ol, ul {{
                margin: 0.8em 0;
                padding-left: 2em;
            }}
            
            li {{
                margin: 0.3em 0;
            }}
            
            code {{
                font-family: 'Courier New', monospace;
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            
            pre {{
                background-color: #f5f5f5;
                padding: 1em;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 9pt;
            }}
            
            .keywords {{
                font-weight: bold;
                margin-top: 1em;
            }}
            
            .page-break {{
                page-break-before: always;
            }}
            
            .references {{
                font-size: 10pt;
            }}
            
            .references ol {{
                padding-left: 1.5em;
            }}
            
            .references li {{
                margin: 0.5em 0;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Process the HTML to add figure captions and improve formatting
    html_template = process_figures(html_template)
    
    # Convert HTML to PDF
    try:
        HTML(string=html_template).write_pdf(output_pdf)
        print(f"Successfully converted {markdown_file} to {output_pdf}")
        return True
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return False

def process_figures(html_content):
    """
    Process figure references and captions in the HTML content
    """
    # Replace figure markdown with proper HTML structure
    figure_pattern = r'!\[([^\]]*)\]\(([^)]+)\)\s*\*([^*]+)\*'
    
    def replace_figure(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        caption = match.group(3)
        
        return f'''
        <div class="figure">
            <img src="{image_path}" alt="{alt_text}" />
            <p class="figure-caption">{caption}</p>
        </div>
        '''
    
    html_content = re.sub(figure_pattern, replace_figure, html_content)
    
    # Add abstract styling
    html_content = html_content.replace('<h2>Abstract</h2>', '<h2>Abstract</h2><div class="abstract">')
    html_content = html_content.replace('<p><strong>Keywords:</strong>', '</div><p class="keywords"><strong>Keywords:</strong>')
    
    # Add page breaks before major sections
    major_sections = ['## 1. Introduction', '## 2. Materials and Methods', '## 3. Results and Discussion', 
                     '## 4. Conclusions', '## References', '## List of Figures', '## Appendix']
    
    for section in major_sections:
        html_content = html_content.replace(f'<h2>{section[3:]}</h2>', f'<div class="page-break"></div><h2>{section[3:]}</h2>')
    
    # Style references section
    html_content = html_content.replace('<h2>References</h2>', '<h2>References</h2><div class="references">')
    html_content = html_content.replace('<h2>List of Figures</h2>', '</div><h2>List of Figures</h2>')
    
    return html_content

def main():
    """Main function to convert the research paper"""
    markdown_file = "research_paper_draft.md"
    output_pdf = "research_paper_draft.pdf"
    
    if not Path(markdown_file).exists():
        print(f"Error: {markdown_file} not found!")
        return
    
    print(f"Converting {markdown_file} to PDF...")
    success = convert_markdown_to_pdf(markdown_file, output_pdf)
    
    if success:
        print(f"PDF created successfully: {output_pdf}")
        print(f"File size: {Path(output_pdf).stat().st_size / 1024:.1f} KB")
    else:
        print("Failed to create PDF")

if __name__ == "__main__":
    main()