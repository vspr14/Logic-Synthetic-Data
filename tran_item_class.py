import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot
import openpyxl
from rdt.transformers.categorical import LabelEncoder

class TranItem:
    
    def __init__(self) -> None:
        self.ratio: float = 0.95
        self.df: pd.DataFrame = pd.DataFrame()
        self.metadata: SingleTableMetadata = SingleTableMetadata()
        self.noisy_metadata: SingleTableMetadata = SingleTableMetadata()
        self.result: list[pd.DataFrame] = [pd.DataFrame(), pd.DataFrame()]
        self.values_with_blank_comment: list[str] = []
        self.model: GaussianCopulaSynthesizer = None
        self.noisy_model: GaussianCopulaSynthesizer = None
        self.constraints: list[dict] = [
            {
                'constraint_class': 'ScalarRange',
                'constraint_parameters': {
                    'column_name': 'DAY',
                    'low_value': 0,
                    'high_value': 32,
                    'strict_boundaries': True
                }
            },
            {
                'constraint_class': 'ScalarRange',
                'constraint_parameters': {
                    'column_name': 'DAY',
                    'low_value': 0,
                    'high_value': 32,
                    'strict_boundaries': True
                }
            },
            {
                'constraint_class': 'UniqueTransactionConstraint',
                'constraint_parameters': {
                    'column_names': ['TRAN_SEQ_NO','ITEM_SEQ_NO']
                }
            },
        ]
        self.le: LabelEncoder = LabelEncoder(add_noise=True)
        self.final: pd.DataFrame = pd.DataFrame()
    
    def reset_metadata(self):
        self.metadata = SingleTableMetadata()
        self.noisy_metadata = SingleTableMetadata()

    def preprocess(self, item_master: str):
        wb = openpyxl.load_workbook(item_master)
        sheet = wb.active
        for cell in sheet[1]:
            if cell.comment and 'blank' in cell.comment.text.lower():
                self.values_with_blank_comment.append(cell.value)
        self.df = self.df.dropna(subset=['ITEM'])
        self.df.dropna(subset=['UNIT_RETAIL'], inplace=True)
        self.values_with_blank_comment.append('CUST_ORDER_NO')
        self.df.drop(self.values_with_blank_comment, axis=1, inplace=True)
        self.df.reset_index(inplace=True, drop=True)

    def set_df(self, df: pd.DataFrame, item_master: str, constraints: bool):
        self.df = df
        self.preprocess(item_master)
        if 'columns' not in self.metadata.to_dict().keys():
            self.metadata.detect_from_dataframe(self.df)
            self.metadata.remove_primary_key()
            self.metadata.update_column("ITEM_SEQ_NO", sdtype="numerical")
            self.metadata.update_column("DAY", sdtype="numerical")
            self.metadata.update_column("QTY", sdtype="numerical")
            self.metadata.update_column("UNIT_RETAIL", sdtype="numerical")
            self.metadata.update_column("ITEM", sdtype="categorical")
            self.metadata.update_column("DEPT", sdtype="categorical")
            self.metadata.update_column("CLASS", sdtype="categorical")
            self.metadata.update_column("SUBCLASS", sdtype="categorical")
            self.metadata.update_column("STORE", sdtype="categorical")
        # self.metadata.save_to_json(filepath='my_metadata_v1.json')
        self.model = GaussianCopulaSynthesizer(metadata=self.metadata)
        self.model.load_custom_constraint_classes(filepath='./constraints/item_constraints.py', class_names=['ItemConstraint','QtyConstraint', 'UniqueTransactionConstraint'])
        self.model.add_constraints(self.constraints)
    
    def add_noise_to_column(self, df: pd.DataFrame, column_name: str, noise_factor: float = 0.1):
        rows = []
        for index, row in df.iterrows():
            original_value = row[column_name]
            noise = np.random.normal(original_value, max(1, abs(original_value) * noise_factor))
            
            if index > 0:
                new_value = abs(original_value * noise + rows[index-1][column_name])
            else:
                new_value = abs(original_value * noise)
            row[column_name] = max(0, int(round(new_value)))
            rows.append(row)
        return pd.DataFrame(rows)

    def gen_sample(self, rows: int, constraints: bool, ratio: float = 0):
        self.ratio = ratio

        if constraints:
            df2: pd.DataFrame = self.add_noise_to_column(self.df, column_name='QTY', noise_factor=100.1)
            df2 = self.add_noise_to_column(df2, column_name='UNIT_RETAIL', noise_factor=3.1)

            self.left = self.df.sample(n=min(5000, self.df.shape[0]))

            self.model.fit(self.left)
            self.result[0] = self.model.sample(rows)

            self.to_noise = df2.sample(n=min(1000, df2.shape[0]))

            # self.after_noise = self.le.fit_transform(self.to_noise, ['DEPT'])
            # self.after_noise = self.le.fit_transform(self.after_noise, ["CLASS"])
            # self.after_noise = self.le.fit_transform(self.after_noise, ["SUBCLASS"])
            if 'columns' not in self.noisy_metadata.to_dict().keys():
                self.noisy_metadata.detect_from_dataframe(self.to_noise)
            self.noisy_metadata.remove_primary_key()
            self.noisy_metadata.update_column("ITEM", sdtype="numerical")
            self.noisy_metadata.update_column("DEPT", sdtype="numerical")
            self.noisy_metadata.update_column("CLASS", sdtype="numerical")
            self.noisy_metadata.update_column("SUBCLASS", sdtype="numerical")
            self.noisy_metadata.update_column("QTY", sdtype="numerical")
            self.noisy_metadata.update_column("UNIT_RETAIL", sdtype="numerical")
            self.noisy_model = GaussianCopulaSynthesizer(self.noisy_metadata)
            self.noisy_model.fit(self.to_noise)
            self.result[1] = self.noisy_model.sample(rows)

            self.final = pd.concat([self.result[0].iloc[:int(self.ratio * rows)], self.result[1].iloc[:int((1-self.ratio) * rows)]])
            self.final.reset_index(inplace=True)
            self.final.drop('index', axis=1, inplace=True)
            self.final['FULFILL_ORDER_NO'] = None
        else:
            self.model.fit(self.df)
            self.final = self.model.sample(rows,max_tries_per_batch=2000)

        self.added_vals = self.final.copy(deep=True)
        for col in self.values_with_blank_comment:
            self.added_vals[col] = None
        return self.added_vals
    
    def metrics(self):
        # if self.df.shape[0] > 5 and self.final.shape[0] > 5:
        #     fig_qty = get_column_plot(self.df, self.final, self.metadata, 'QTY')
        #     fig_qty.data[0].marker.color = 'red'
        #     fig_retail = get_column_plot(self.df, self.final, self.metadata, 'UNIT_RETAIL')
        #     fig_retail.data[0].marker.color = 'red'
        # else:
        #     fig_qty = fig_retail = "No metrics available for this dataset."
        metrics_string = "Number of rows generated: "+str(self.final.shape[0])+"  \nOverall Quality Score: "+str(round((evaluate_quality(self.df, self.final, self.metadata).get_score())*100,2))+"%"
        unique_df = pd.DataFrame(columns=['Attribute', 'Invalid', 'Valid'])
        unique_df.loc[0] = ['ITEM', self.final[~self.final['ITEM'].isin(self.df['ITEM'])].shape[0], self.final[self.final['ITEM'].isin(self.df['ITEM'])].shape[0]]
        unique_df.loc[1] = ['DEPT', self.final[~self.final['DEPT'].isin(self.df['DEPT'])].shape[0], self.final[self.final['DEPT'].isin(self.df['DEPT'])].shape[0]]
        unique_df.loc[2] = ['CLASS', self.final[~self.final['CLASS'].isin(self.df['CLASS'])].shape[0], self.final[self.final['CLASS'].isin(self.df['CLASS'])].shape[0]]
        unique_df.loc[3] = ['SUBCLASS', self.final[~self.final['SUBCLASS'].isin(self.df['SUBCLASS'])].shape[0], self.final[self.final['SUBCLASS'].isin(self.df['SUBCLASS'])].shape[0]]
        return metrics_string, unique_df