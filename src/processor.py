import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu


class SepsisDataProcessor:
    """
    Handles preprocessing and analysis of the MIMIC-III sepsis dataset.
    Covers label creation, temporal window extraction, variable filtering,
    train/test splitting, tensor conversion, and standardization.
    """

    def __init__(self, file_path):
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)

    def initial_prep(self):
        """
        Creates patient-level labels and sepsis onset hour from raw SepsisLabel column.
        Adds will_have_sepsis and sepsis_onset_hour columns if not already present.
        """
        if 'will_have_sepsis' not in self.df.columns:
            if 'SepsisLabel' in self.df.columns:
                sepsis_info = self.df.groupby('Patient_ID')['SepsisLabel'].max().reset_index()
                sepsis_info.columns = ['Patient_ID', 'will_have_sepsis']
                self.df = self.df.merge(sepsis_info, on='Patient_ID', how='left')

        if 'sepsis_onset_hour' not in self.df.columns:
            if 'SepsisLabel' in self.df.columns:
                sepsis_onset = self.df[self.df['SepsisLabel'] == 1].groupby('Patient_ID')['Hour'].min().reset_index()
                sepsis_onset.columns = ['Patient_ID', 'sepsis_onset_hour']
                self.df = self.df.merge(sepsis_onset, on='Patient_ID', how='left')

        return self.df.copy()

    def window_selection(self,
                         window_start: int = -12,
                         window_end: int = 0,
                         start_for_non_sepsis: bool = True,
                         non_sepsis_ratio: float = 1.0):
        """
        Selects a fixed-length temporal window around sepsis onset for each patient.
        Window bounds are inclusive: [window_start, window_end] relative to onset.
        Non-sepsis patients are sampled from the beginning of their stay by default.
        """
        window_size = (window_end - window_start) + 1

        if window_size <= 0:
            raise ValueError("window_end must be greater than or equal to window_start.")

        final_patients = []

        sepsis_df = self.df[self.df['will_have_sepsis'] == 1]

        for pid, group in sepsis_df.groupby('Patient_ID'):
            onset = group['sepsis_onset_hour'].iloc[0]
            abs_start = onset + window_start
            abs_end = onset + window_end
            window_data = group[(group['Hour'] >= abs_start) & (group['Hour'] <= abs_end)]
            if len(window_data) == window_size:
                final_patients.append(window_data)

        sepsis_count = len(final_patients)
        print(f"Sepsis patients retained (window [{window_start}, {window_end}] inclusive): {sepsis_count}")

        non_sepsis_df = self.df[self.df['will_have_sepsis'] == 0]
        valid_non_sepsis = []

        for pid, group in non_sepsis_df.groupby('Patient_ID'):
            total_hours = len(group)
            if total_hours >= window_size:
                if start_for_non_sepsis:
                    valid_non_sepsis.append(group.head(window_size))
                else:
                    start_idx = np.random.randint(0, total_hours - window_size + 1)
                    valid_non_sepsis.append(group.iloc[start_idx: start_idx + window_size])

        target_non_sepsis_count = int(sepsis_count * non_sepsis_ratio)
        if len(valid_non_sepsis) > target_non_sepsis_count:
            indices = np.random.choice(len(valid_non_sepsis), target_non_sepsis_count, replace=False)
            valid_non_sepsis = [valid_non_sepsis[i] for i in indices]

        print(f"Non-sepsis patients retained (ratio {non_sepsis_ratio}): {len(valid_non_sepsis)}")

        if final_patients or valid_non_sepsis:
            final_df = pd.concat(final_patients + valid_non_sepsis).reset_index(drop=True)
        else:
            final_df = pd.DataFrame(columns=self.df.columns)

        self.df = final_df.copy()

        return final_df

    def filter_variables(self, df_input=None, min_patient_presence_pct: float = 0.80, essential_cols=None):
        """
        Removes columns that have no measured value for more than (1 - min_patient_presence_pct)
        of patients. Essential columns are always kept regardless of coverage.
        """
        if df_input is None:
            df_input = self.df.copy()

        presence_per_patient = df_input.groupby('Patient_ID').apply(lambda x: x.notna().any())
        patient_coverage = presence_per_patient.mean()
        cols_to_keep = patient_coverage[patient_coverage >= min_patient_presence_pct].index.tolist()

        if essential_cols is None:
            essential_cols = ['Patient_ID', 'Hour', 'will_have_sepsis', 'Lactate',
                              'Creatinine', 'MAP', 'SBP', 'Bilirubin_total', 'Platelets',
                              'Temp', 'WBC']

        final_cols = list(set(cols_to_keep + essential_cols))

        print(f"Variables retained ({min_patient_presence_pct * 100}% presence threshold): "
              f"{len(final_cols)} out of {df_input.shape[1]}")
        self.df = df_input[final_cols].copy()
        return df_input[final_cols]

    def filter_physio_variables(self, df_input=None, is_hour=False):
        """
        Drops administrative, demographic, and technical columns.
        Retains only physiological signals, lab values, and will_have_sepsis.
        """
        if not is_hour:
            cols_to_drop = [
                'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime',
                'sepsis_onset_hour', 'ICULOS', 'Unnamed: 0', 'Unnamed: 0.1', 'SepsisLabel']
        else:
            cols_to_drop = [
                'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime',
                'ICULOS', 'Unnamed: 0', 'Unnamed: 0.1', 'SepsisLabel']

        if df_input is None:
            df_input = self.df.copy()

        existing_drop = [c for c in cols_to_drop if c in df_input.columns]
        df_cleaned = df_input.drop(columns=existing_drop)

        print(f"Non-physiological variables removed: {len(existing_drop)} columns dropped.")
        self.df = df_cleaned.copy()
        return df_cleaned

    def split_train_test(self, df_input=None, test_size=0.2, random_state=42):
        """
        Splits data into train and test sets at the patient level to prevent leakage.
        Stratification is applied to preserve the sepsis/non-sepsis ratio in both sets.
        """
        if df_input is None:
            df_input = self.df

        unique_patients = df_input['Patient_ID'].unique()
        labels = df_input.groupby('Patient_ID')['will_have_sepsis'].first().values

        train_ids, test_ids = train_test_split(
            unique_patients,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        df_train = df_input[df_input['Patient_ID'].isin(train_ids)]
        df_test = df_input[df_input['Patient_ID'].isin(test_ids)]

        print(f"Train set: {df_train['Patient_ID'].nunique()} patients")
        print(f"Test set:  {df_test['Patient_ID'].nunique()} patients")

        return df_train, df_test

    def to_tensor(self, df_filtered=None, exclude_cols=None):
        """
        Converts the filtered DataFrame into 3D tensors suitable for model input.
        Output X has shape (N_patients, T_timesteps, C_features), y has shape (N_patients,).
        """
        if exclude_cols is None:
            exclude_cols = ['Patient_ID', 'Hour', 'will_have_sepsis']

        if df_filtered is None:
            df_filtered = self.df.copy()

        df_sorted = df_filtered.sort_values(['Patient_ID', 'Hour'])
        y = df_sorted.groupby('Patient_ID')['will_have_sepsis'].first().values
        features_df = df_sorted.drop(columns=exclude_cols)

        n_patients = df_sorted['Patient_ID'].nunique()
        n_hours = df_sorted.groupby('Patient_ID').size().iloc[0]
        n_features = features_df.shape[1]

        x = features_df.values.reshape(n_patients, n_hours, n_features)

        print(f"Tensor X shape: {x.shape} (patients, hours, features)")
        print(f"Vector y shape: {y.shape}")

        return x, y

    @staticmethod
    def standardize_tensors(x_train, x_test):
        """
        Standardizes 3D tensors (N, T, C) by fitting a StandardScaler on the training set
        and applying the same transformation to the test set to prevent data leakage.
        """
        n_tr, t_tr, c_tr = x_train.shape
        n_te, t_te, c_te = x_test.shape

        scaler = StandardScaler()

        x_train_flat = x_train.reshape(-1, c_tr)
        x_test_flat = x_test.reshape(-1, c_te)

        print(f"Standardizing {c_tr} features...")

        x_train_scaled_flat = scaler.fit_transform(x_train_flat)
        x_test_scaled_flat = scaler.transform(x_test_flat)

        x_train_scaled = x_train_scaled_flat.reshape(n_tr, t_tr, c_tr)
        x_test_scaled = x_test_scaled_flat.reshape(n_te, t_te, c_te)

        print("Standardization complete.")

        return x_train_scaled, x_test_scaled

    @staticmethod
    def plot_sepsis_onset_distribution(df_input):
        """Plots the distribution of sepsis onset hour across all sepsis patients."""
        onsets = df_input.groupby('Patient_ID')['sepsis_onset_hour'].first().dropna()
        plt.figure(figsize=(10, 6))
        sns.histplot(onsets, bins=30, kde=True, color='red')
        plt.title("Distribution of Sepsis Onset Hour")
        plt.xlabel("Hour (from admission)")
        plt.ylabel("Number of patients")
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_nan_stats(df_input, top_n=15):
        """Plots the top variables with the highest percentage of missing values."""
        nan_percent = (df_input.isna().sum() / len(df_input)) * 100
        nan_percent = nan_percent.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        nan_percent.plot(kind='bar', color='skyblue')
        plt.title(f"Top {top_n} Variables by Missing Value Rate (%)")
        plt.ylabel("% NaN")
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_feature_evolution(df_input, feature_name):
        """
        Plots the temporal evolution of a physiological feature aligned on sepsis onset (T=0).
        Sepsis and non-sepsis patients are overlaid with a data completeness background.
        """
        cols_tech = ['Patient_ID', 'will_have_sepsis', 'Hour', 'sepsis_onset_hour']
        if feature_name in cols_tech:
            return

        df_plot = df_input.copy()
        df_plot['Time_To_Onset'] = df_plot['Hour'] - df_plot['sepsis_onset_hour']

        mask_sain = df_plot['will_have_sepsis'] == 0
        if mask_sain.any():
            mean_offset = df_plot[df_plot['will_have_sepsis'] == 1]['Time_To_Onset'].min()
            if np.isnan(mean_offset):
                mean_offset = 0
            df_plot.loc[mask_sain, 'Time_To_Onset'] = (
                df_plot[mask_sain].groupby('Patient_ID').cumcount() + mean_offset
            )

        completeness = df_plot.groupby('Time_To_Onset')[feature_name].apply(lambda x: x.notna().mean())

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax2.fill_between(completeness.index, 0, completeness.values,
                         color='gray', alpha=0.1, label='Data completeness')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Data completeness (0 to 1)")

        sns.lineplot(
            data=df_plot, x='Time_To_Onset', y=feature_name, hue='will_have_sepsis',
            palette={0: '#3498db', 1: '#e74c3c'}, marker='o', ax=ax1, errorbar='sd'
        )

        ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Sepsis onset (T=0)')

        title_suffix = "Pre-Sepsis" if df_plot['Time_To_Onset'].max() <= 0 else "Around Sepsis"
        ax1.set_title(f"Evolution of {feature_name} - {title_suffix}")
        ax1.set_xlabel("Hours relative to diagnosis (0 = Onset)")
        ax1.set_ylabel(f"{feature_name}")

        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='upper left')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mortality_vs_sepsis(df_input):
        """Plots the proportion of in-hospital mortality across sepsis and non-sepsis patients."""
        if 'HospMortality' in df_input.columns:
            patient_stats = df_input.groupby('Patient_ID').agg({
                'will_have_sepsis': 'max',
                'HospMortality': 'max'
            })
            ct = pd.crosstab(patient_stats['will_have_sepsis'], patient_stats['HospMortality'], normalize='index')
            ct.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'])
            plt.title("Mortality Rate: Sepsis vs Non-Sepsis")
            plt.xlabel("Sepsis (0=No, 1=Yes)")
            plt.ylabel("Proportion")
            plt.legend(["Survival", "Death"], title="Outcome")
            plt.show()
        else:
            print("Column 'HospMortality' not found in dataset.")

    from scipy.stats import mannwhitneyu

    @staticmethod
    def quantify_differences(df, feature):
        """Runs a Mann-Whitney U test to assess whether a feature differs significantly between groups."""
        sepsis = df[df['will_have_sepsis'] == 1][feature].dropna()
        sain = df[df['will_have_sepsis'] == 0][feature].dropna()

        stat, p_value = mannwhitneyu(sepsis, sain)
        print(f"Feature: {feature} | P-value: {p_value:.5f}")
        if p_value < 0.05:
            print("Statistically significant difference detected.")
