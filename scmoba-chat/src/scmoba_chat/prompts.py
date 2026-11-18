"""Exact Chat training instructions; default questions are legacy training templates.

Source: src/dataset/natural_template_instructions_Chat.py and muti_dataset_v4_sex.py.
No automatic task inference from arbitrary free text. Non-free tasks start independently.
"""
TASKS = {
    'celltype': {
        'label': 'Broad cell class',
        'instruction': 'Classify this expression profile into its broad cell class and give a brief answer.',
        'question': 'Can you predict the cell type based on this gene expression data? Please provide your answer.'},
    'subclass': {
        'label': 'Cell subclass',
        'instruction': 'Classify this expression profile into its fine-grained cell subclass and give a brief answer.',
        'question': 'Can you predict the cell subtype based on this gene expression data? Please provide your answer.'},
    'tissuetype': {
        'label': 'Tissue / brain region',
        'instruction': 'Identify the tissue or brain region represented by this expression profile and give a brief answer.',
        'question': 'Predict the tissue type based on this gene expression data.'},
    'agegroup': {
        'label': 'Annotated age group',
        'instruction': 'Identify the annotated age group associated with this expression profile and give a brief answer.',
        'question': 'What is the most likely developmental stage of this sample based on its gene expression?'},
    'developmentstage': {
        'label': 'Age / developmental-stage label',
        'instruction': 'Identify the developmental stage associated with this expression profile and give a brief answer.',
        'question': 'What is the most likely age of this sample based on its gene expression?'},
    'disease': {
        'label': 'Disease / health label (research only)',
        'instruction': 'Identify the disease or health state associated with this expression profile and give a brief answer.',
        'question': 'Predict the disease using this transcriptomic data.'},
    'sex': {
        'label': 'Biological sex',
        'instruction': 'Identify the biological sex associated with this expression profile and give a brief answer.',
        'question': 'What is the biological sex associated with this single-cell profile?'},
}
FREE_QUESTION = "Provide a detailed overview of this cell's characteristics and potential molecular context."


def task_question(task='free', question=None):
    if task=='free':
        if not isinstance(question,str) or not question.strip():raise ValueError('Free QA requires a question')
        return question, question
    if task not in TASKS:raise ValueError(f'Unsupported task: {task}')
    original=TASKS[task]['question'] if question is None or not question.strip() else question
    if any(original.startswith(t['instruction']) for t in TASKS.values()):
        raise ValueError('Supply only the original question; the Chat instruction is added automatically')
    return original, TASKS[task]['instruction']+'\n\n'+original


def catalog():
    return {'free':{'label':'Free-text conversation','question':FREE_QUESTION,'instruction':''},**TASKS}
