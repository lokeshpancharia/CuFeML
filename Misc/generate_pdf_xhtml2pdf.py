import os
from xhtml2pdf import pisa

def convert_html_to_pdf(source_html, output_filename):
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    
    source_path = os.path.join(project_dir, source_html)
    output_path = os.path.join(project_dir, output_filename)

    print(f"Reading HTML from: {source_path}")
    
    # Open input file
    with open(source_path, "r", encoding='utf-8') as f:
        source_html_content = f.read()

    # Open output file
    with open(output_path, "w+b") as result_file:
        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            source_html_content,                # the HTML to convert
            dest=result_file,                   # file handle to recieve result
            encoding='utf-8'
        )

    # return True on success and False on errors
    if pisa_status.err:
        print(f"Error generating PDF: {pisa_status.err}")
        return False
    else:
        print(f"Successfully created PDF: {output_path}")
        return True

if __name__ == "__main__":
    try:
        convert_html_to_pdf('research_paper_formal.html', 'research_paper_formal_xhtml2pdf.pdf')
    except ImportError:
        print("xhtml2pdf is not installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
