#!/usr/bin/env python
"""
Check registration quality report.
"""
import json
from pathlib import Path

report_path = Path('preprocessed_registered/registration_report.json')

if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)
    
    print('=' * 70)
    print('REGISTRATION QUALITY REPORT')
    print('=' * 70)
    
    correlations = []
    for subject in report:
        if 'T1w' in subject.get('modalities', {}):
            corr = subject['modalities']['T1w'].get('correlation', 0)
            correlations.append(corr)
            
            # Status indicator
            if corr > 0.7:
                status = 'GOOD'
            elif corr > 0.5:
                status = 'REVIEW'
            else:
                status = 'POOR'
            
            print(f"{subject['subject']}: {corr:.3f} - {status}")
    
    if correlations:
        avg = sum(correlations) / len(correlations)
        print('=' * 70)
        print(f'Average correlation: {avg:.3f}')
        
        good = sum(1 for c in correlations if c > 0.7)
        print(f'Good registrations (>0.7): {good}/{len(correlations)}')
        
        print('=' * 70)
        
        if avg > 0.7:
            print('\nEXCELLENT! Ready to train!')
            print('All registrations are high quality.')
        elif avg > 0.5:
            print('\nACCEPTABLE quality.')
            print('Review subjects with low correlation before training.')
        else:
            print('\nPOOR quality - check your data!')
            print('Registration may have failed. Investigate the issues.')
        
        print('\nNext step: Train the model')
        print('Run: python scripts/train_gan_enhanced.py --epochs 10 --batch-size 2')
else:
    print('ERROR: Registration report not found!')
    print(f'Expected location: {report_path.absolute()}')
    print('\nDid registration complete successfully?')
