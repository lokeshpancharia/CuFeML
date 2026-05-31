import markdown
import markdown
import os

def convert_paper():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from Misc to CuFeML
    project_dir = os.path.dirname(base_dir)
    input_md = os.path.join(project_dir, 'research_paper_formal.md')
    output_html = os.path.join(project_dir, 'research_paper_formal.html')
    output_pdf = os.path.join(project_dir, 'research_paper_formal.pdf')

    print(f"Reading from: {input_md}")

    with open(input_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to HTML
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Add some basic styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Research Paper</title>
        <style>
            body {{ font-family: "Times New Roman", serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f5f5f5; padding: 2px 4px; }}
            .caption {{ text-align: center; font-style: italic; color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """

    # Save HTML
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Created HTML: {output_html}")

    # Convert to PDF
    print("Generating PDF...")
    try:
        from weasyprint import HTML
        # We need to ensure image paths are correct for weasyprint
        # The HTML has relative paths like "Misc/Figure_1.png"
        # WeasyPrint resolves relative to the base_url (which defaults to CWD)
        # We should set base_url to the project directory
        HTML(string=html_content, base_url=project_dir).write_pdf(output_pdf)
        print(f"Created PDF: {output_pdf}")
    except Exception as e:
        print(f"Could not generate PDF due to missing dependencies: {e}")
        print(f"Please open {output_html} in your browser and select 'Print to PDF'.")

if __name__ == "__main__":
    convert_paper()
