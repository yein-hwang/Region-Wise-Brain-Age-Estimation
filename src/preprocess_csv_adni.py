import pandas as pd
import os
import glob

# 데이터프레임 로드
group = 'ad'
base_csv = pd.read_csv(f'/media/leelabsg-storage1/yein/research/data/csv/fsdat_baseline_{group}.csv')
cols = ['File_name', 'PTAGE', 'PTGENDER', 'Group.at.enrollment']
root = '/media/leelabsg-storage1/yein/research/data/adni_region/disease'

def mk_csv(df):
    df = df.copy()  # 원본 데이터프레임의 사본을 생성
    df.rename(columns={
        'File_name':'subjectID', 
        'PTAGE': 'age',
        'PTGENDER': 'gender',
        'Group.at.enrollment': 'group'
        }, inplace=True)

    csv_data = {
        'subjectID': [],
        'imgs': [],
        'mask': [],
        'age': [],
        'gender': [],
        'group': []
    }

    regions = ['caudate', 'cerebellum', 'frontal_lobe', 'insula', 'occipital_lobe', 'parietal_lobe', 'putamen', 'temporal_lobe', 'thalamus']
    for region in regions:
        csv_data[region] = []
        csv_data[f'{region}_mask'] = []

    for i in df.index:
        subject = df.loc[i, 'subjectID']
        
        csv_data['subjectID'].append(subject)
        csv_data['age'].append(df.loc[i, 'age'])
        csv_data['gender'].append(df.loc[i, 'gender'])
        csv_data['group'].append(df.loc[i, 'group'])
        csv_data['imgs'].append(f'{root}/{subject}/T1w_registered.nii.gz')
        csv_data['mask'].append(f'{root}/{subject}/T1w_brain_mask_registered.nii.gz')
        csv_data['caudate'].append(f'{root}/{subject}/region_1.nii.gz')
        csv_data['caudate_mask'].append(f'{root}/{subject}/region_1_mask.nii.gz')
        csv_data['cerebellum'].append(f'{root}/{subject}/region_2.nii.gz')
        csv_data['cerebellum_mask'].append(f'{root}/{subject}/region_2_mask.nii.gz')
        csv_data['frontal_lobe'].append(f'{root}/{subject}/region_3.nii.gz')
        csv_data['frontal_lobe_mask'].append(f'{root}/{subject}/region_3_mask.nii.gz')
        csv_data['insula'].append(f'{root}/{subject}/region_4.nii.gz')
        csv_data['insula_mask'].append(f'{root}/{subject}/region_4_mask.nii.gz')
        csv_data['occipital_lobe'].append(f'{root}/{subject}/region_5.nii.gz')
        csv_data['occipital_lobe_mask'].append(f'{root}/{subject}/region_5_mask.nii.gz')
        csv_data['parietal_lobe'].append(f'{root}/{subject}/region_6.nii.gz')
        csv_data['parietal_lobe_mask'].append(f'{root}/{subject}/region_6_mask.nii.gz')
        csv_data['putamen'].append(f'{root}/{subject}/region_7.nii.gz')
        csv_data['putamen_mask'].append(f'{root}/{subject}/region_7_mask.nii.gz')
        csv_data['temporal_lobe'].append(f'{root}/{subject}/region_8.nii.gz')
        csv_data['temporal_lobe_mask'].append(f'{root}/{subject}/region_8_mask.nii.gz')
        csv_data['thalamus'].append(f'{root}/{subject}/region_9.nii.gz')
        csv_data['thalamus_mask'].append(f'{root}/{subject}/region_9_mask.nii.gz')

    new_df = pd.DataFrame(csv_data)
    return new_df

# Save the csv file by each disease group
def group_csv(df):

    groups = df['Group.at.enrollment'].unique().tolist()
    group_map = {}
    for i in range(len(groups)):
        group_map.setdefault(i, groups[i])

    for _, v in group_map.items():

        adni_reg_csv = df[df['Group.at.enrollment'] == v][cols]
        print(f"Current Group Dataset: {v}, {len(adni_reg_csv)}")

        new_df = mk_csv(adni_reg_csv)

        new_df.to_csv(f'/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/adni_{v}_region.csv', index=False)

# MCI 값을 포함하고 있는 모든 컬럼을 찾는 함수
def get_columns_with_value(df, value):
    columns_with_value = []
    for col in df.columns:
        if df[col].astype(str).str.contains(value).any():
            columns_with_value.append(col)
    return columns_with_value

# MCI 값을 포함하는 컬럼 추출
# columns_with_mci = get_columns_with_value(base_csv, 'MCI')
# df_mci = base_csv[base_csv[columns_with_mci].apply(lambda row: row.astype(str).str.contains('MCI').any(), axis=1)][cols]

df_ = base_csv[cols]

new_df = mk_csv(df_)
new_df.to_csv(f'/media/leelabsg-storage1/yein/research/BAE/RegionBAE/data/adni_{group}_region.csv', index=False)