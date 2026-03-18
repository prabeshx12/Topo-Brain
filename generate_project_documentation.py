"""
Generate comprehensive project documentation as Word file
Covers: ADNI preprocessing + inference pipeline
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime

def add_heading_style(doc, text, level, color=None):
    """Add heading with optional color"""
    heading = doc.add_heading(text, level=level)
    if color:
        for run in heading.runs:
            run.font.color.rgb = RGBColor(*color)
    return heading

def add_table_with_data(doc, headers, rows):
    """Add formatted table"""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        header_cells[i].text = header
        header_cells[i].paragraphs[0].runs[0].font.bold = True
    
    # Data rows
    for row_idx, row_data in enumerate(rows, 1):
        cells = table.rows[row_idx].cells
        for col_idx, value in enumerate(row_data):
            cells[col_idx].text = str(value)
    
    return table

def main():
    # Create document
    doc = Document()
    
    # ======================== TITLE PAGE ========================
    add_heading_style(doc, 'Topo-Brain: ADNI Diffusion Model', 0, (0, 102, 204))
    add_heading_style(doc, 'Complete Processing & Inference Pipeline', 2)
    
    doc.add_paragraph()
    title_para = doc.add_paragraph('Full Workflow Documentation')
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.runs[0].font.size = Pt(14)
    
    doc.add_paragraph()
    doc.add_paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y")}')
    doc.add_paragraph('Project: Topo-Brain - 3T→7T MRI Super-Resolution via Diffusion Models')
    doc.add_paragraph('Focus: ADNI Alzheimer\'s Disease Dataset Processing')
    doc.add_paragraph('Author: Prabesh X')
    doc.add_paragraph('Repository: https://github.com/prabeshx12/Topo-Brain')
    
    doc.add_page_break()
    
    # ======================== TABLE OF CONTENTS ========================
    add_heading_style(doc, 'Table of Contents', 1)
    toc_items = [
        '1. Project Overview',
        '2. Architecture Design',
        '3. Dataset Information',
        '4. Component Implementation',
        '5. Step 1: BIDS Conversion',
        '6. Step 2: ADNI-Specific Preprocessing',
        '7. Step 3: Full-Volume Inference',
        '8. Step 4: Quality Control & Diagnostics',
        '9. Configuration Files',
        '10. Kaggle Notebook Workflow',
        '11. Git Management',
        '12. Troubleshooting & Next Steps'
    ]
    for item in toc_items:
        doc.add_paragraph(item, style='List Number')
    
    doc.add_page_break()
    
    # ======================== 1. PROJECT OVERVIEW ========================
    add_heading_style(doc, '1. Project Overview', 1, (0, 102, 204))
    
    doc.add_heading('1.1 Objective', level=2)
    doc.add_paragraph(
        'Deploy a trained diffusion-based 3T→7T brain MRI synthesis model on the ADNI '
        '(Alzheimer\'s Disease Neuroimaging Initiative) dataset containing 200 unpaired 3T brains.'
    )
    
    doc.add_heading('1.2 Challenge', level=2)
    doc.add_paragraph(
        'Extend a model trained on 10 healthy subjects (paired 3T-7T) to 200 ADNI patients with:'
    )
    doc.add_paragraph('Different population (Alzheimer\'s Disease vs healthy)', style='List Bullet')
    doc.add_paragraph('Brain atrophy and anatomical changes', style='List Bullet')
    doc.add_paragraph('Older scanner protocols with field inhomogeneities', style='List Bullet')
    doc.add_paragraph('Variable voxel spacing (0.9-1.5mm vs ~1.0mm)', style='List Bullet')
    doc.add_paragraph('Unpaired data (3T only, no 7T reference)', style='List Bullet')
    
    doc.add_heading('1.3 Solution Architecture', level=2)
    doc.add_paragraph(
        'Three-stage pipeline ensuring domain adaptation and inference compatibility:'
    )
    doc.add_paragraph('Stage 1: BIDS Conversion (Raw ADNI → standardized BIDS format)', style='List Number')
    doc.add_paragraph('Stage 2: Preprocessing (compatible with training pipeline)', style='List Number')
    doc.add_paragraph('Stage 3: Inference (full-volume 3T→7T synthesis)', style='List Number')
    
    doc.add_page_break()
    
    # ======================== 2. ARCHITECTURE DESIGN ========================
    add_heading_style(doc, '2. Architecture Design', 1, (0, 102, 204))
    
    doc.add_heading('2.1 Overall System Architecture', level=2)
    doc.add_paragraph(
        'The pipeline consists of three integrated components working in sequence:'
    )
    
    # Create architecture table
    arch_data = [
        ['STAGE', 'INPUT', 'PROCESSING', 'OUTPUT', 'KEY COMPONENT'],
        ['Stage 1: Conversion', 'Raw ADNI (nested folders)', 'BIDS mapping', 'BIDS structure', 'convert_adni_to_bids.py'],
        ['Stage 2: Preprocessing', 'BIDS data', 'N4 + skull strip + normalize', 'Preprocessed 3T', 'preprocess_bids.py'],
        ['Stage 3: Inference', 'Preprocessed 3T', 'Tiled diffusion (64³)', 'Synthetic 7T + seg', 'infer_adni_batch.py'],
    ]
    table = doc.add_table(rows=len(arch_data), cols=len(arch_data[0]))
    table.style = 'Light Grid Accent 1'
    
    # Header
    for i, header in enumerate(arch_data[0]):
        table.rows[0].cells[i].text = header
        table.rows[0].cells[i].paragraphs[0].runs[0].font.bold = True
    
    # Data
    for row_idx, row_data in enumerate(arch_data[1:], 1):
        for col_idx, value in enumerate(row_data):
            table.rows[row_idx].cells[col_idx].text = value
    
    doc.add_heading('2.2 Data Flow Diagram', level=2)
    doc.add_paragraph(
        'Raw ADNI Data (002_S_1018/.../file.nii)'
        '  ↓'
        'convert_adni_to_bids.py'
        '  ↓'
        'BIDS Structure (sub-002S1018/ses-20061129/anat/*_T1w.nii.gz)'
        '  ↓'
        'preprocess_bids.py (with preprocess_adni.yaml)'
        '  ↓'
        'Preprocessed 3T (N4-corrected, skull-stripped, normalized [-1,1])'
        '  ↓'
        'infer_adni_batch.py'
        '  ↓'
        'Synthetic 7T + Segmentation'
        '  ↓'
        'analyze_inference_diagnostics.py'
        '  ↓'
        'Risk Reports & Quality Metrics'
    )
    
    doc.add_heading('2.3 Model Architecture', level=2)
    doc.add_paragraph('Core Model: AnatomyGuidedUNet (Conditional 3D U-Net with dual decoders)')
    
    model_spec = [
        ['Component', 'Specification'],
        ['Architecture', '3D U-Net with ResNet-style blocks'],
        ['Encoders', 'Anatomy-aware feature extraction from 3T brain'],
        ['Decoders', 'Dual: (1) 7T synthesis, (2) Tissue segmentation'],
        ['Diffusion Steps', '200 timesteps (vs DDPM\'s 1000)'],
        ['Beta Schedule', 'Cosine schedule for smooth noise progression'],
        ['Loss Functions', 'L1 (reconstruction) + perceptual (VGG) + segmentation'],
        ['Patch Size', '64³ voxels'],
        ['Training Subjects', '10 subjects (20 volumes: 3T-7T pairs)'],
    ]
    table = add_table_with_data(doc, model_spec[0], model_spec[1:])
    
    doc.add_heading('2.4 Key Features', level=2)
    doc.add_paragraph('Tiled Inference: Full-volume processing via 64³ patches with 32-voxel overlap', style='List Bullet')
    doc.add_paragraph('Tukey Window Blending: Smooth patch transitions at boundaries', style='List Bullet')
    doc.add_paragraph('Brain Masking: Automatic skull stripping via HD-BET', style='List Bullet')
    doc.add_paragraph('Variable Dimensions: Supports ADNI\'s 240-280×180-220×120-180 voxel ranges', style='List Bullet')
    doc.add_paragraph('Quality Control: Per-subject diagnostics + risk scoring', style='List Bullet')
    
    doc.add_page_break()
    
    # ======================== 3. DATASET INFORMATION ========================
    add_heading_style(doc, '3. Dataset Information', 1, (0, 102, 204))
    
    doc.add_heading('3.1 Training Dataset (Healthy Subjects)', level=2)
    training_spec = [
        ['Property', 'Value'],
        ['Subjects', '10 healthy controls'],
        ['Sessions per subject', '2 (3T + 7T)'],
        ['Total volumes', '20 (10 3T + 10 7T)'],
        ['Modalities', 'T1w, T2w'],
        ['Voxel spacing', '~1.0mm³ isotropic'],
        ['Image dimensions', '~256×256×166 voxels'],
        ['Brain state', 'Normal anatomy'],
        ['Use case', 'Training paired 3T→7T synthesis model'],
    ]
    add_table_with_data(doc, training_spec[0], training_spec[1:])
    
    doc.add_heading('3.2 ADNI Dataset (Alzheimer\'s Disease)', level=2)
    adni_spec = [
        ['Property', 'Value'],
        ['Subjects', '200 ADNI participants'],
        ['Sessions per subject', 'Variable (1-4 visits)'],
        ['Total volumes', '~389 3T scans (estimated)'],
        ['Modalities', 'T1w (primary)'],
        ['Voxel spacing', 'Variable (0.9-1.5mm)'],
        ['Image dimensions', 'Variable (240-280 × 180-220 × 120-180)'],
        ['Brain state', 'Alzheimer\'s Disease with atrophy'],
        ['Scanner age', 'Older protocols (pre-2015)'],
        ['Use case', 'Inference on unpaired 3T only'],
    ]
    add_table_with_data(doc, adni_spec[0], adni_spec[1:])
    
    doc.add_heading('3.3 Key Dataset Differences', level=2)
    diff_spec = [
        ['Aspect', 'Training', 'ADNI', 'Impact'],
        ['Brain atrophy', 'None', 'Significant', 'Requires lower skull strip threshold (0.4 vs 0.5)'],
        ['Voxel spacing', 'Uniform ~1mm', 'Variable', 'No resampling, keep native for inference'],
        ['Scanner quality', 'Modern', 'Legacy', 'Need stricter N4 bias correction'],
        ['Data pairing', 'Paired (3T-7T)', 'Unpaired (3T only)', 'Inference mode only'],
        ['Population', 'Healthy', 'AD patients', 'Domain shift mitigation via preprocessing'],
    ]
    add_table_with_data(doc, diff_spec[0], diff_spec[1:])
    
    doc.add_page_break()
    
    # ======================== 4. COMPONENT IMPLEMENTATION ========================
    add_heading_style(doc, '4. Component Implementation', 1, (0, 102, 204))
    
    doc.add_heading('4.1 Created Scripts', level=2)
    doc.add_paragraph(
        'Three new scripts were developed and integrated into the repository: '
        'feat/adni-inference-preprocess-integration branch'
    )
    
    doc.add_heading('4.1.1 scripts/convert_adni_to_bids.py', level=3)
    doc.add_paragraph(
        'Purpose: Transform ADNI raw folder structure to BIDS-compliant layout'
    )
    
    convert_spec = [
        ['Property', 'Details'],
        ['Input format', 'ADNI: 002_S_1018/MPR__GradWarp/.../ADNI_002_S_1018_MR_*.nii'],
        ['Output format', 'BIDS: sub-002S1018/ses-20061129/anat/*_T1w.nii.gz'],
        ['Key functions', 'to_bids_subject(), date_to_session(), find_nii_files(), convert()'],
        ['Execution result', 'Converted 389 NIfTI files (0 skipped)'],
        ['Outputs', 'participants.tsv, dataset_description.json, conversion_report.csv'],
    ]
    add_table_with_data(doc, convert_spec[0], convert_spec[1:])
    
    doc.add_paragraph('Example command:')
    code_para = doc.add_paragraph(
        'python scripts/convert_adni_to_bids.py '
        '--input-root MRI-AD-Part-2/ADNI '
        '--output-root MRI-AD-Part-2/ADNI_BIDS'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('4.1.2 scripts/infer_adni_batch.py', level=3)
    doc.add_paragraph(
        'Purpose: Full-volume tiled diffusion inference on ADNI 3T images'
    )
    
    infer_spec = [
        ['Property', 'Details'],
        ['Input', 'Preprocessed 3T T1w images (variable dimensions)'],
        ['Processing', 'Tiled 64³ patches with 32-voxel overlap + Tukey windowing'],
        ['Output', '*_desc-synth7T_T1w.nii.gz (synthetic 7T), *_desc-synth7Tseg_dseg.nii.gz (segmentation)'],
        ['Diagnostics', 'Per-subject JSON with intensity range, shape, inference status'],
        ['Brain masking', 'Automatic skull strip during preprocessing, cleanup post-inference'],
        ['Expected runtime', '3-5 min per subject (GPU dependent)'],
    ]
    add_table_with_data(doc, infer_spec[0], infer_spec[1:])
    
    doc.add_paragraph('Example command:')
    code_para = doc.add_paragraph(
        'python scripts/infer_adni_batch.py '
        '--checkpoint models/checkpoint_141000.pt '
        '--train-config configs/train_diffusion.yaml '
        '--preprocessed-root MRI-AD-Part-2/ADNI_BIDS/derivatives/topobrain-preproc-adni '
        '--output-root adni_inference_results '
        '--limit 1'  # Start with 1 for testing
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('4.1.3 scripts/analyze_inference_diagnostics.py', level=3)
    doc.add_paragraph(
        'Purpose: Analyze inference outputs and generate risk reports'
    )
    
    diag_spec = [
        ['Property', 'Details'],
        ['Input', 'Inference output directory with *_inference_diagnostics.json files'],
        ['Risk criteria', 'Intensity mismatch, small dimensions, voxel anisotropy, failures'],
        ['Risk levels', 'critical, high, medium, low'],
        ['Output CSV', 'diagnostics_risk_report.csv (sorted by risk_score)'],
        ['Output JSON', 'diagnostics_risk_report.json (with summary statistics)'],
    ]
    add_table_with_data(doc, diag_spec[0], diag_spec[1:])
    
    doc.add_paragraph('Example command:')
    code_para = doc.add_paragraph(
        'python scripts/analyze_inference_diagnostics.py '
        '--input adni_inference_results'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_page_break()
    
    # ======================== 5. STEP 1: BIDS CONVERSION ========================
    add_heading_style(doc, '5. Step 1: BIDS Conversion (Raw ADNI → BIDS)', 1, (0, 102, 204))
    
    doc.add_heading('5.1 Overview', level=2)
    doc.add_paragraph(
        'Raw ADNI data is organized in complex nested folders. '
        'This step converts to standardized BIDS format for reproducibility and automation.'
    )
    
    doc.add_heading('5.2 Input Structure', level=2)
    input_struct = """ADNI/
├── 002_S_1018/
│   ├── MPR__GradWarp__B1_Correction__N3__Scaled/
│   │   └── 2006-11-29_10_00_05.0/
│   │       └── I40817/
│   │           └── ADNI_002_S_1018_MR_....nii
│   └── (other sequences)
├── 002_S_1319/
│   └── (similar structure)
└── ... (389 total subjects)"""
    code_para = doc.add_paragraph(input_struct)
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('5.3 Output Structure (BIDS)', level=2)
    output_struct = """ADNI_BIDS/
├── dataset_description.json
├── participants.tsv
├── conversion_report.csv
├── sub-002S1018/
│   └── ses-20061129/
│       └── anat/
│           ├── sub-002S1018_ses-20061129_T1w.nii.gz
│           └── sub-002S1018_ses-20061129_T1w.json
├── sub-002S1319/
│   └── ses-YYYYMMDD/
│       └── anat/
└── ... (389 T1w images total)"""
    code_para = doc.add_paragraph(output_struct)
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('5.4 Conversion Logic', level=2)
    doc.add_paragraph('Subject ID extraction: Regex pattern from ADNI_XXX_S_XXXX format', style='List Bullet')
    doc.add_paragraph('Date parsing: Extract timestamp from folder name → session ID', style='List Bullet')
    doc.add_paragraph('File filtering: Select only MRI files (ADNI_*_MR_*.nii)', style='List Bullet')
    doc.add_paragraph('Metadata: Generate JSON sidecars with sequence info', style='List Bullet')
    
    doc.add_heading('5.5 Execution', level=2)
    doc.add_paragraph('Command:')
    code_para = doc.add_paragraph(
        'python scripts/convert_adni_to_bids.py '
        '--input-root MRI-AD-Part-2/ADNI '
        '--output-root MRI-AD-Part-2/ADNI_BIDS'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_paragraph('Result: Converted 389 files, 0 skipped')
    doc.add_paragraph('Verification: 389 T1w images + 389 JSON sidecars ✓')
    
    doc.add_page_break()
    
    # ======================== 6. STEP 2: PREPROCESSING ========================
    add_heading_style(doc, '6. Step 2: ADNI-Specific Preprocessing', 1, (0, 102, 204))
    
    doc.add_heading('6.1 Overview', level=2)
    doc.add_paragraph(
        'Preprocessing ensures ADNI data is compatible with the trained inference model. '
        'Critical: Must preserve normalization, spacing, and patch logic from training.'
    )
    
    doc.add_heading('6.2 Preprocessing Pipeline', level=2)
    doc.add_paragraph('Stage 1: N4 Bias Field Correction (SimpleITK)', style='List Number')
    doc.add_paragraph(
        '  ├─ Removes low-frequency intensity artifacts from legacy ADNI scanners',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  ├─ ADNI params: 50 iterations, convergence=0.0001 (stricter than training)',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  └─ Output: N4-corrected image',
        style='List Bullet 2'
    )
    
    doc.add_paragraph('Stage 2: Skull Stripping (HD-BET)', style='List Number')
    doc.add_paragraph(
        '  ├─ Removes non-brain tissue (skull, meninges, etc.)',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  ├─ ADNI threshold: 0.4 (vs 0.5 for healthy) to preserve atrophic gray matter',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  └─ Output: Skull-stripped brain + binary mask',
        style='List Bullet 2'
    )
    
    doc.add_paragraph('Stage 3: Intensity Normalization', style='List Number')
    doc.add_paragraph(
        '  ├─ Normalize to [-1, 1] range (required by diffusion model)',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  ├─ Uses percentile-based method (0.5–99.5) to preserve brain contrast',
        style='List Bullet 2'
    )
    doc.add_paragraph(
        '  └─ IDENTICAL to training preprocessing',
        style='List Bullet 2'
    )
    
    doc.add_heading('6.3 Configuration File: preprocess_adni.yaml', level=2)
    doc.add_paragraph('Location: configs/preprocess_adni.yaml')
    
    config_params = [
        ['Parameter', 'Value', 'Reason'],
        ['skull_strip.bet_threshold', '0.4', 'Lower for atrophic brains'],
        ['n4_convergence_threshold', '0.0001', 'Stricter for old scanner artifacts'],
        ['normalization.method', 'diffusion', 'IDENTICAL to training'],
        ['normalization.percentile', '0.5–99.5', 'IDENTICAL to training'],
        ['resample.target_spacing', 'null', 'CRITICAL: Keep native spacing for inference'],
    ]
    add_table_with_data(doc, config_params[0], config_params[1:])
    
    doc.add_heading('6.4 Compatibility Guarantee', level=2)
    compat_table = [
        ['Component', 'Training', 'ADNI (Kaggle)', 'Match?'],
        ['Normalization', '[-1, 1]', '[-1, 1]', '✓ IDENTICAL'],
        ['Voxel spacing', 'Native', 'Native', '✓ IDENTICAL'],
        ['Patch size', '64³', '64³', '✓ IDENTICAL'],
        ['N4 bias correction', 'Enabled', 'Enabled*', '✓ Compatible (stricter)'],
        ['Skull strip method', 'HD-BET', 'HD-BET*', '✓ Compatible (threshold=0.4)'],
        ['Resampling', 'None', 'None', '✓ IDENTICAL'],
    ]
    add_table_with_data(doc, compat_table[0], compat_table[1:])
    doc.add_paragraph('*ADNI-specific tuning; essential for domain adaptation')
    
    doc.add_heading('6.5 Execution on Kaggle', level=2)
    doc.add_paragraph('Use the provided Kaggle notebook (notebooks/kaggle_preprocessing.ipynb)')
    doc.add_paragraph('Will automatically:')
    doc.add_paragraph('1. Clone Topo-Brain repository', style='List Number')
    doc.add_paragraph('2. Install dependencies (nibabel, MONAI, HD-BET, SimpleITK)', style='List Number')
    doc.add_paragraph('3. Run preprocessing with preprocess_adni.yaml config', style='List Number')
    doc.add_paragraph('4. Output to: derivatives/topobrain-preproc-adni/', style='List Number')
    
    doc.add_page_break()
    
    # ======================== 7. STEP 3: INFERENCE ========================
    add_heading_style(doc, '7. Step 3: Full-Volume Inference', 1, (0, 102, 204))
    
    doc.add_heading('7.1 Overview', level=2)
    doc.add_paragraph(
        'Run diffusion-based 3T→7T synthesis on all 389 preprocessed ADNI images. '
        'Uses tiled inference to handle variable image dimensions.'
    )
    
    doc.add_heading('7.2 Tiled Inference Architecture', level=2)
    doc.add_paragraph(
        'Challenge: ADNI images have variable dimensions (not fixed to 64³). '
        'Solution: Extract overlapping 64³ patches, blend at boundaries.'
    )
    
    tiling_spec = [
        ['Parameter', 'Value', 'Purpose'],
        ['Patch size', '64³ voxels', 'Match training input size'],
        ['Overlap', '32 voxels', 'Enable smooth boundary blending'],
        ['Blend function', 'Tukey window', 'Smooth transitions (avoids artifacts)'],
        ['Brain masking', 'Post-inference cleanup', 'Remove out-of-brain noise'],
    ]
    add_table_with_data(doc, tiling_spec[0], tiling_spec[1:])
    
    doc.add_heading('7.3 Algorithm: Tiled Inference with Windowing', level=2)
    doc.add_paragraph('For each preprocessed 3T image:')
    doc.add_paragraph('1. Load full-volume image (e.g., 256×256×166)', style='List Number')
    doc.add_paragraph('2. Define sliding window: 64³ patches, 32-voxel stride', style='List Number')
    doc.add_paragraph('3. For each patch:', style='List Number')
    doc.add_paragraph('   a. Extract patch from 3T volume', style='List Bullet 2')
    doc.add_paragraph('   b. Run diffusion inference → synthetic 7T patch + segmentation', style='List Bullet 2')
    doc.add_paragraph('   c. Apply Tukey window for smooth blending', style='List Bullet 2')
    doc.add_paragraph('   d. Accumulate into output volume (weighted average)', style='List Bullet 2')
    doc.add_paragraph('4. Apply final brain mask (cleanup edges)', style='List Number')
    doc.add_paragraph('5. Save synthetic 7T + segmentation maps', style='List Number')
    
    doc.add_heading('7.4 Workflow', level=2)
    doc.add_paragraph('Inputs: Preprocessed ADNI 3T (from Step 2)')
    doc.add_paragraph('Outputs: For each subject:')
    doc.add_paragraph('• {subject}_{session}_desc-synth7T_T1w.nii.gz (synthetic 7T)', style='List Bullet')
    doc.add_paragraph('• {subject}_{session}_desc-synth7Tseg_dseg.nii.gz (tissue segmentation)', style='List Bullet')
    doc.add_paragraph('• {subject}_{session}_inference_diagnostics.json (quality metrics)', style='List Bullet')
    doc.add_paragraph('• inference_summary.json (batch-level stats)', style='List Bullet')
    
    doc.add_heading('7.5 Execution', level=2)
    doc.add_paragraph('Command (local or CERNBox):')
    code_para = doc.add_paragraph(
        'python scripts/infer_adni_batch.py \\\n'
        '  --checkpoint models/checkpoint_141000.pt \\\n'
        '  --train-config configs/train_diffusion.yaml \\\n'
        '  --preprocessed-root MRI-AD-Part-2/ADNI_BIDS/derivatives/topobrain-preproc-adni \\\n'
        '  --output-root adni_inference_results \\\n'
        '  --patch-size 64 \\\n'
        '  --overlap 32'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('7.6 Expected Performance', level=2)
    perf_spec = [
        ['Metric', 'Expected Value'],
        ['Runtime per subject', '3–5 minutes (GPU: V100/A100)'],
        ['Total runtime (389 subjects)', '20–40 hours GPU time'],
        ['Memory per subject', '6–8 GB VRAM'],
        ['Output size per subject', '50–100 MB (3 files)'],
        ['Total output storage', '~20–40 GB'],
    ]
    add_table_with_data(doc, perf_spec[0], perf_spec[1:])
    
    doc.add_page_break()
    
    # ======================== 8. STEP 4: QUALITY CONTROL ========================
    add_heading_style(doc, '8. Step 4: Quality Control & Diagnostics', 1, (0, 102, 204))
    
    doc.add_heading('8.1 Per-Subject Diagnostics', level=2)
    doc.add_paragraph(
        'Each inference outputs a {subject}_{session}_inference_diagnostics.json file with:'
    )
    
    diag_fields = [
        ['Field', 'Meaning'],
        ['status', '"ok" or "error"'],
        ['input_shape', 'Original 3T dimensions'],
        ['input_spacing', 'Voxel sizes'],
        ['intensity_range', '[min, max] of input'],
        ['in_expected_range', 'True if [-1.2, 1.2]'],
        ['output_shape', 'Synthetic 7T dimensions'],
        ['error_message', 'If status="error"'],
    ]
    add_table_with_data(doc, diag_fields[0], diag_fields[1:])
    
    doc.add_heading('8.2 Risk Scoring', level=2)
    doc.add_paragraph(
        'Run analyze_inference_diagnostics.py to generate risk reports:'
    )
    
    doc.add_paragraph('Command:')
    code_para = doc.add_paragraph(
        'python scripts/analyze_inference_diagnostics.py '
        '--input adni_inference_results'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_paragraph('Risk levels (based on multiple criteria):')
    risk_levels = [
        ['Level', 'Criteria', 'Action'],
        ['CRITICAL', 'Inference failed OR dimensions < 64 voxels', 'Manual review required'],
        ['HIGH', 'Intensity anomaly (out of [-1.2, 1.2]) OR strong anisotropy', 'Verify preprocessing'],
        ['MEDIUM', 'Minor issues (e.g., edge-case anisotropy)', 'Monitor during validation'],
        ['LOW', 'All checks pass', 'Suitable for clinical analysis'],
    ]
    add_table_with_data(doc, risk_levels[0], risk_levels[1:])
    
    doc.add_heading('8.3 Output Reports', level=2)
    doc.add_paragraph('diagnostics_risk_report.csv: All subjects ranked by risk_score')
    doc.add_paragraph('diagnostics_risk_report.json: Risk distribution + summary statistics')
    doc.add_paragraph('Example summary: "389 subjects: 350 LOW, 35 MEDIUM, 4 HIGH, 0 CRITICAL"')
    
    doc.add_heading('8.4 Visual QC (Optional)', level=2)
    doc.add_paragraph(
        'For each preprocessed image, visualize:'
    )
    doc.add_paragraph('Slice 1: Original 3T image', style='List Bullet')
    doc.add_paragraph('Slice 2: Brain mask overlay', style='List Bullet')
    doc.add_paragraph('Slice 3: Intensity histogram (confirm [-1, 1] range)', style='List Bullet')
    
    doc.add_page_break()
    
    # ======================== 9. CONFIGURATION FILES ========================
    add_heading_style(doc, '9. Configuration Files', 1, (0, 102, 204))
    
    doc.add_heading('9.1 preprocess_adni.yaml', level=2)
    doc.add_paragraph('Location: configs/preprocess_adni.yaml')
    doc.add_paragraph('Purpose: Configure preprocessing for ADNI data')
    
    config_content = """data_root: data
output_root: derivatives/topobrain-preproc-adni
modalities:
  - T1w
session_3t: ses-1
prefer_aligned: false
require_aligned: false
include_derivatives: false
output_suffix: desc-preproc
overwrite: false
resume: true
seed: 42

skull_strip:
  method: hd-bet
  device: cuda
  mode: accurate
  bet_threshold: 0.4  # ← ADNI: lower for atrophy
  use_existing_masks: true

bias_correction:
  enabled: true
  n4_iterations: 50
  n4_convergence_threshold: 0.0001  # ← ADNI: stricter

normalization:
  method: diffusion
  percentile_lower: 0.5
  percentile_upper: 99.5
  clip_lower_percentile: null
  clip_upper_percentile: null

resample:
  target_spacing: null  # ← CRITICAL: Keep native spacing
  interpolation: linear

qc:
  enabled: true
  max_samples: 10
  output_dir: null"""
    
    code_para = doc.add_paragraph(config_content)
    code_para.paragraph_format.left_indent = Inches(0.5)
    code_para.style = 'List Bullet'
    
    doc.add_heading('9.2 train_diffusion.yaml (Already in repo)', level=2)
    doc.add_paragraph('Location: configs/train_diffusion.yaml')
    doc.add_paragraph('Used by: infer_adni_batch.py to load model configuration')
    doc.add_paragraph('Key parameters:')
    doc.add_paragraph('• model.patch_size: 64 (matches tiled inference)', style='List Bullet')
    doc.add_paragraph('• model.timesteps: 200 (diffusion steps)', style='List Bullet')
    doc.add_paragraph('• model.beta_schedule: "cosine" (noise schedule)', style='List Bullet')
    
    doc.add_page_break()
    
    # ======================== 10. KAGGLE NOTEBOOK ========================
    add_heading_style(doc, '10. Kaggle Notebook Workflow', 1, (0, 102, 204))
    
    doc.add_heading('10.1 Purpose', level=2)
    doc.add_paragraph(
        'Provide reproducible preprocessing on Kaggle for users without local GPU setup.'
    )
    
    doc.add_heading('10.2 Notebook: kaggle_preprocessing.ipynb', level=2)
    doc.add_paragraph('Location: notebooks/kaggle_preprocessing.ipynb')
    
    notebook_structure = [
        ['Cell', 'Purpose'],
        ['1', 'Mount Kaggle datasets and setup paths'],
        ['2', 'Install dependencies (nibabel, MONAI, HD-BET, SimpleITK)'],
        ['3', 'Clone Topo-Brain repository'],
        ['4', 'Discover T1w BIDS files'],
        ['5A', 'Display ADNI preprocessing configuration'],
        ['5B', 'Run standard preprocessing with preprocess_adni.yaml'],
        ['6', 'Verify preprocessed output (count files, directory structure)'],
        ['7', 'Generate summary report (manifest, statistics)'],
        ['8', 'Verify inference compatibility (intensity range, spacing, masking)'],
        ['9', 'Summary and next steps'],
    ]
    add_table_with_data(doc, notebook_structure[0], notebook_structure[1:])
    
    doc.add_heading('10.3 Usage Instructions', level=2)
    doc.add_paragraph('on Kaggle:')
    doc.add_paragraph('1. Click "New Notebook"', style='List Number')
    doc.add_paragraph('2. Select "Code" (Python)', style='List Number')
    doc.add_paragraph('3. Copy notebook cells sequentially', style='List Number')
    doc.add_paragraph('4. Update DATA_ROOT path to match your uploaded dataset', style='List Number')
    doc.add_paragraph('5. Run cells in order', style='List Number')
    
    doc.add_heading('10.4 Data Upload to Kaggle', level=2)
    doc.add_paragraph('Dataset to upload: MRI-AD-Part-2/ADNI_BIDS (389 T1w images + JSON sidecars)')
    doc.add_paragraph('Size: ~50–100 GB')
    doc.add_paragraph('Format: BIDS (output from Step 1)')
    doc.add_paragraph('Alternative: Upload raw ADNI + run conversion step inside notebook')
    
    doc.add_page_break()
    
    # ======================== 11. GIT MANAGEMENT ========================
    add_heading_style(doc, '11. Git Management & Repository Structure', 1, (0, 102, 204))
    
    doc.add_heading('11.1 Branch Structure', level=2)
    doc.add_paragraph('Repository: https://github.com/prabeshx12/Topo-Brain')
    doc.add_paragraph('Active branch: feat/adni-inference-preprocess-integration')
    doc.add_paragraph('Default branch: main (unchanged, only code/docs updates)')
    
    doc.add_heading('11.2 New Files Added', level=2)
    new_files = [
        ['File', 'Purpose', 'Lines'],
        ['scripts/convert_adni_to_bids.py', 'BIDS converter', '202'],
        ['scripts/infer_adni_batch.py', 'Tiled inference', '489'],
        ['scripts/analyze_inference_diagnostics.py', 'Risk analyzer', '145'],
        ['configs/preprocess_adni.yaml', 'ADNI preprocessing config', '70'],
        ['notebooks/kaggle_preprocessing.ipynb', 'Kaggle workflow', '10 cells'],
    ]
    add_table_with_data(doc, new_files[0], new_files[1:])
    
    doc.add_heading('11.3 .gitignore Updates', level=2)
    doc.add_paragraph('Added to prevent pushing large dataset files:')
    
    gitignore_content = """MRI-AD-Part-2/
MRI-AD-Part-2/ADNI/
MRI-AD-Part-2/ADNI_BIDS/
MRI-AD-Part-2/adni_inference_results/
MRI-AD-Part-2/**/derivatives/"""
    
    code_para = doc.add_paragraph(gitignore_content)
    code_para.paragraph_format.left_indent = Inches(0.5)
    code_para.style = 'List Bullet'
    
    doc.add_paragraph('Effect: Dataset completely ignored from git (only code/docs tracked)')
    
    doc.add_heading('11.4 Repository Status', level=2)
    doc.add_paragraph('Untracked (visible to git):')
    doc.add_paragraph('• M .gitignore', style='List Bullet')
    doc.add_paragraph('• M scripts/train_diffusion.py (if modified)', style='List Bullet')
    doc.add_paragraph('• ?? scripts/convert_adni_to_bids.py ← NEW', style='List Bullet')
    doc.add_paragraph('• ?? scripts/infer_adni_batch.py ← NEW', style='List Bullet')
    doc.add_paragraph('• ?? scripts/analyze_inference_diagnostics.py ← NEW', style='List Bullet')
    doc.add_paragraph('• ?? configs/preprocess_adni.yaml ← NEW', style='List Bullet')
    doc.add_paragraph('• ?? notebooks/kaggle_preprocessing.ipynb ← NEW', style='List Bullet')
    
    doc.add_paragraph('Hidden from git (in .gitignore):')
    doc.add_paragraph('❌ MRI-AD-Part-2/ (all dataset files)', style='List Bullet')
    
    doc.add_page_break()
    
    # ======================== 12. TROUBLESHOOTING ========================
    add_heading_style(doc, '12. Troubleshooting & Common Issues', 1, (0, 102, 204))
    
    doc.add_heading('12.1 BIDS Conversion Issues', level=2)
    
    doc.add_heading('Issue: No files converted (Converted: 0)', level=3)
    doc.add_paragraph('Cause: Input folder structure doesn\'t match ADNI format')
    doc.add_paragraph('Solution: Verify folder naming convention (002_S_1018/.../ADNI_*.nii)')
    
    doc.add_heading('12.2 Preprocessing Issues', level=2)
    
    doc.add_heading('Issue: HD-BET fails / timeout', level=3)
    doc.add_paragraph('Cause: GPU memory or installation issue')
    doc.add_paragraph('Solution:')
    doc.add_paragraph('• Fallback to CPU: set device="cpu" in config', style='List Bullet')
    doc.add_paragraph('• Verify HD-BET installation: pip install HD-BET --force-reinstall', style='List Bullet')
    
    doc.add_heading('Issue: CUDA out of memory', level=3)
    doc.add_paragraph('Cause: GPU VRAM insufficient')
    doc.add_paragraph('Solution:')
    doc.add_paragraph('• Use CPU-only mode (slower)', style='List Bullet')
    doc.add_paragraph('• Process one subject at a time', style='List Bullet')
    doc.add_paragraph('• Reduce batch size if applicable', style='List Bullet')
    
    doc.add_heading('12.3 Inference Issues', level=2)
    
    doc.add_heading('Issue: "No preprocessed T1w inputs found"', level=3)
    doc.add_paragraph('Cause: --preprocessed-root path incorrect')
    doc.add_paragraph('Solution: Verify output directory from preprocessing step')
    
    doc.add_heading('Issue: Low inference quality / artifacts in output', level=3)
    doc.add_paragraph('Cause: Domain shift (ADNI vs. training data)')
    doc.add_paragraph('Solution:')
    doc.add_paragraph('• Verify skull strip threshold is 0.4 (for atrophy)', style='List Bullet')
    doc.add_paragraph('• Check intensity normalization is [-1, 1]', style='List Bullet')
    doc.add_paragraph('• Review diagnostics report for risk warnings', style='List Bullet')
    
    doc.add_heading('12.4 Performance Issues', level=2)
    
    doc.add_heading('Issue: Very slow inference (>10 min/subject)', level=3)
    doc.add_paragraph('Cause: CPU-only or old GPU')
    doc.add_paragraph('Solution: Use modern GPU (V100, A100)'  )
    
    doc.add_page_break()
    
    # ======================== SUMMARY ========================
    add_heading_style(doc, 'Complete Workflow Summary', 1, (0, 102, 204))
    
    doc.add_heading('End-to-End Pipeline', level=2)
    
    summary_steps = [
        ['Phase', 'Script/Tool', 'Input', 'Output', 'Time'],
        ['Phase 1: Conversion', 'convert_adni_to_bids.py', 'Raw ADNI', 'BIDS (389 files)', '5–10 min'],
        ['Phase 2: Preprocessing', 'preprocess_bids.py', 'BIDS T1w', 'Preprocessed 3T', '2–4 hours'],
        ['Phase 3a: Inference (Sample)', 'infer_adni_batch.py (--limit 1)', '1 preprocessed 3T', '1 synthetic 7T', '3–5 min'],
        ['Phase 3b: Inference (Full)', 'infer_adni_batch.py', '389 preprocessed 3T', '389 synthetic 7T', '20–40 hours GPU'],
        ['Phase 4: QC', 'analyze_inference_diagnostics.py', 'Inference outputs', 'Risk reports', '5–10 min'],
    ]
    add_table_with_data(doc, summary_steps[0], summary_steps[1:])
    
    doc.add_heading('Key Deliverables', level=2)
    doc.add_paragraph('✓ 389 BIDS-formatted 3T brain images', style='List Bullet')
    doc.add_paragraph('✓ 389 preprocessed (N4, skull-stripped, normalized) 3T images', style='List Bullet')
    doc.add_paragraph('✓ 389 synthetic 7T-like images (via diffusion inference)', style='List Bullet')
    doc.add_paragraph('✓ 389 tissue segmentation maps', style='List Bullet')
    doc.add_paragraph('✓ Per-subject quality metrics & risk scores', style='List Bullet')
    doc.add_paragraph('✓ Comprehensive diagnostics report', style='List Bullet')
    
    doc.add_heading('Next Steps', level=2)
    doc.add_paragraph('1. Test on single subject (--limit 1) to verify setup', style='List Number')
    doc.add_paragraph('2. Run full batch inference once validated', style='List Number')
    doc.add_paragraph('3. Review risk reports and manual QC', style='List Number')
    doc.add_paragraph('4. Perform clinical validation with radiologists', style='List Number')
    doc.add_paragraph('5. Analyze synthesized 7T for downstream tasks', style='List Number')
    
    doc.add_page_break()
    
    # ======================== APPENDIX ========================
    add_heading_style(doc, 'Appendix: Command Reference', 1, (0, 102, 204))
    
    doc.add_heading('A1. BIDS Conversion', level=2)
    code_para = doc.add_paragraph(
        'python scripts/convert_adni_to_bids.py \\\n'
        '  --input-root MRI-AD-Part-2/ADNI \\\n'
        '  --output-root MRI-AD-Part-2/ADNI_BIDS'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('A2. Preprocessing (Local)', level=2)
    code_para = doc.add_paragraph(
        'python scripts/preprocess_bids.py \\\n'
        '  --config configs/preprocess_adni.yaml \\\n'
        '  --data-root MRI-AD-Part-2/ADNI_BIDS \\\n'
        '  --output-root MRI-AD-Part-2/ADNI_BIDS/derivatives/topobrain-preproc-adni'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('A3. Inference (Single Subject Test)', level=2)
    code_para = doc.add_paragraph(
        'python scripts/infer_adni_batch.py \\\n'
        '  --checkpoint ~/checkpoints/checkpoint_141000.pt \\\n'
        '  --train-config configs/train_diffusion.yaml \\\n'
        '  --preprocessed-root MRI-AD-Part-2/ADNI_BIDS/derivatives/topobrain-preproc-adni \\\n'
        '  --output-root adni_inference_results \\\n'
        '  --limit 1'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('A4. Inference (Full Batch)', level=2)
    code_para = doc.add_paragraph(
        'python scripts/infer_adni_batch.py \\\n'
        '  --checkpoint ~/checkpoints/checkpoint_141000.pt \\\n'
        '  --train-config configs/train_diffusion.yaml \\\n'
        '  --preprocessed-root MRI-AD-Part-2/ADNI_BIDS/derivatives/topobrain-preproc-adni \\\n'
        '  --output-root adni_inference_results'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_heading('A5. Quality Control & Risk Analysis', level=2)
    code_para = doc.add_paragraph(
        'python scripts/analyze_inference_diagnostics.py \\\n'
        '  --input adni_inference_results'
    )
    code_para.paragraph_format.left_indent = Inches(0.5)
    
    # Save document
    output_path = 'D:\\11PrabeshX\\Projects\\latest\\Topo-Brain\\Topo-Brain_ADNI_Complete_Documentation.docx'
    doc.save(output_path)
    
    return output_path

if __name__ == '__main__':
    path = main()
    print(f"✓ Document created: {path}")
