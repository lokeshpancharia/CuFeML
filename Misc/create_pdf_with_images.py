#!/usr/bin/env python3
"""
Create PDF with embedded images using reportlab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import markdown
from pathlib import Path
import re
import os

def create_pdf_with_images(markdown_file, output_pdf):
    """
    Create PDF from markdown file with embedded images
    """
    
    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Times-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        fontName='Times-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=16,
        fontName='Times-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Times-Roman',
        leading=14
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Times-Italic',
        leading=12
    )
    
    abstract_style = ParagraphStyle(
        'Abstract',
        parent=body_style,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=20,
        spaceBefore=10
    )
    
    # Build story (content)
    story = []
    
    # Parse markdown content into sections
    sections = parse_markdown_with_images(markdown_content)
    
    for section in sections:
        if section['type'] == 'title':
            story.append(Paragraph(section['content'], title_style))
            story.append(Spacer(1, 12))
            
        elif section['type'] == 'h1':
            story.append(Paragraph(section['content'], heading1_style))
            
        elif section['type'] == 'h2':
            story.append(Paragraph(section['content'], heading2_style))
            
        elif section['type'] == 'abstract':
            story.append(Paragraph("Abstract", heading1_style))
            story.append(Paragraph(section['content'], abstract_style))
            
        elif section['type'] == 'paragraph':
            story.append(Paragraph(section['content'], body_style))
            
        elif section['type'] == 'table':
            table = create_table(section['content'])
            if table:
                story.append(table)
                story.append(Spacer(1, 12))
                
        elif section['type'] == 'image':
            # Add the actual image
            image_path = section['path']
            if os.path.exists(image_path):
                try:
                    # Calculate image size to fit page
                    img = Image(image_path)
                    img_width, img_height = img.imageWidth, img.imageHeight
                    
                    # Scale to fit page width (max 6 inches)
                    max_width = 6 * inch
                    max_height = 4 * inch
                    
                    if img_width > max_width:
                        scale = max_width / img_width
                        img_width = max_width
                        img_height = img_height * scale
                    
                    if img_height > max_height:
                        scale = max_height / img_height
                        img_height = max_height
                        img_width = img_width * scale
                    
                    img.drawWidth = img_width
                    img.drawHeight = img_height
                    
                    story.append(Spacer(1, 12))
                    story.append(img)
                    story.append(Paragraph(section['caption'], caption_style))
                    story.append(Spacer(1, 12))
                    
                except Exception as e:
                    print(f"Error adding image {image_path}: {e}")
                    story.append(Paragraph(f"[Figure: {section['caption']}]", body_style))
            else:
                story.append(Paragraph(f"[Figure not found: {section['caption']}]", body_style))
                
        elif section['type'] == 'page_break':
            story.append(PageBreak())
    
    # Build PDF
    try:
        doc.build(story)
        print(f"Successfully created PDF with images: {output_pdf}")
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False

def parse_markdown_with_images(content):
    """
    Parse markdown content with proper image handling
    """
    sections = []
    lines = content.split('\n')
    current_paragraph = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('# '):
            # Title
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'title',
                'content': line[2:]
            })
            
        elif line.startswith('## '):
            # H1
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'h1',
                'content': line[3:]
            })
            
        elif line.startswith('### '):
            # H2
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'h2',
                'content': line[4:]
            })
            
        elif line.startswith('| '):
            # Table
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
            
            # Parse table
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1  # Back up one line
            
            sections.append({
                'type': 'table',
                'content': table_lines
            })
            
        elif line.startswith('!['):
            # Image with caption
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
            
            # Extract image info
            match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if match:
                caption_text = match.group(1)
                image_path = match.group(2)
                
                # Check if next line has caption in italics
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('*') and lines[i + 1].strip().endswith('*'):
                    caption_text = lines[i + 1].strip()[1:-1]  # Remove asterisks
                    i += 1  # Skip the caption line
                
                sections.append({
                    'type': 'image',
                    'caption': caption_text,
                    'path': image_path
                })
            
        elif line == '':
            # Empty line - end current paragraph
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
                current_paragraph = []
                
        else:
            # Regular text
            if line and not line.startswith('*Figure') and not line.startswith('*Table'):
                current_paragraph.append(line)
        
        i += 1
    
    # Add final paragraph if exists
    if current_paragraph:
        sections.append({
            'type': 'paragraph',
            'content': ' '.join(current_paragraph)
        })
    
    return sections

def create_table(table_lines):
    """
    Create a reportlab table from markdown table lines
    """
    if len(table_lines) < 2:
        return None
    
    # Parse table data
    data = []
    for line in table_lines:
        if '---' in line:  # Skip separator line
            continue
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
        if cells:
            data.append(cells)
    
    if not data:
        return None
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    return table

def main():
    """Main function"""
    markdown_file = "research_paper_draft.md"
    output_pdf = "research_paper_with_images.pdf"
    
    if not Path(markdown_file).exists():
        print(f"Error: {markdown_file} not found!")
        return
    
    print(f"Creating PDF with images from {markdown_file}...")
    success = create_pdf_with_images(markdown_file, output_pdf)
    
    if success:
        print(f"PDF with images created successfully: {output_pdf}")
        print(f"File size: {Path(output_pdf).stat().st_size / 1024:.1f} KB")
    else:
        print("Failed to create PDF")

if __name__ == "__main__":
    main()