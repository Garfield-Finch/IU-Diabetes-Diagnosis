import pandas as pd
import os


class DataProcess:
    ls_categorical_features = [
        "s_female",
        "s_poverty",
        "s_race",
        "s_educ",
        "s_married",
        "s_private",
        "s_medicaid",
        "s_access",
        "s_smoke4",
        "s_alcohol",
        "phy_score",
        "gestational_diabetes",
    ]

    ls_numerical_features = [
        "s_age",
        "d_totaldietadjs",
        "s_sleep",
        "familydm",
        "l_bmxbmi",  # BMI
    ]

    ls_lab_features = [
        "l_lbxtc",  # Total Cholesterol (mg/dL)
        "l_lbxtr",  # Triglyceride (mg/dL)
        "l_lbxgh",  # Glycohemoglobin(%) (can be 0 according NHANES)
        "l_lbxglu",  # Glycohemoglobin(%) (can be 0 according NHANES)
        "l_meansbp",  # Mean Systolic BP (can be 0 according NHANES)
        "l_meandbp",  # Mean Diastolic BP  (can be 0 according NHANES)
        "l_ldlnew",  # Known LDL LDL=Total cholesterol-HDL-trigl/5 (only works when trig <=400)
        "l_dminsulin",  # Insulin (pmol/L)
        "l_bu",  # Blood Urea Nitrogen (mg/dL)
        "l_ua",  # Albumin, urine (mg/L)
        "l_cr",  # Creatinine, urine (mg/dL)
    ]

    ls_demographic_features = [
        "s_age",
        "s_female",
        "s_race",
        "s_married",
    ]

    ls_body_features = [
        "l_bmxbmi",
        "l_bmxwt",
        "l_bmxwaist",
    ]

    ls_ada_self_test_features = [
        "s_age",
        "s_female",
        "l_meandbp",  # ADA self test asks whether one has high blood pressure
        "s_race",
        "l_bmxbmi",
        "familydm",
        "gestational_diabetes",
        "phy_score",
    ]

    ls_medical_features = [
        "l_cholmed",
        "l_dmmed",
        "l_bpmed",
        "l_dmoral",
    ]

    ls_health_status_features = [
        "hx_htn",
        "phy_score",
        # "phy_score_year",
        "l_dmrisk",
        "gestational_diabetes",
    ]

    ls_family_history_features = [
        "familydm",
        "familydm_meanSBP",
        "familydm_meanDBP",
        "familydm_BPMed",
    ]

    ls_health_behavior_features = [
        "s_smoke",
        "s_smoke4",
        "s_alcohol",
        "s_sleep",
        "d_totaldietadjs",
    ]

    ls_social_features = [
        "s_poverty",
        "s_educ",
    ]

    ls_healthcare_features = [
        "s_insurance",
        "s_insgov",
        "s_medicaid",
        "s_private",
        "s_access",
    ]

    # ## encoded features
    ls_encoded_features = [
        "s_female_0",
        "s_female_1",
        "s_poverty_0",
        "s_poverty_1",
        "s_poverty_2",
        "s_poverty_3",
        "s_race_0",
        "s_race_1",
        "s_race_2",
        "s_race_3",
        "s_educ_0",
        "s_educ_1",
        "s_educ_2",
        "s_educ_3",
        "s_married_0",
        "s_married_1",
        "s_married_2",
        "s_private_0",
        "s_private_1",
        "s_medicaid_0",
        "s_medicaid_1",
        "s_access_0",
        "s_access_1",
        "s_access_2",
        "s_smoke4_0",
        "s_smoke4_1",
        "s_smoke4_2",
        "s_smoke4_3",
        "s_alcohol_0",
        "s_alcohol_1",
        "s_alcohol_2",
        "phy_score_0",
        "phy_score_1",
        "phy_score_2",
        "phy_score_255",
        "gestational_diabetes_0",
        "gestational_diabetes_1",
        "gestational_diabetes_255",
    ]

    ls_ada_score = [
        "ada_familydm",
        "ada_physical",
        "ada_bpmed",
        "ada_overall",
        "ada_age",
        "ada_gender",
        "ada_bmi",
    ]

    # ## The previous features are basic features grouped in their categories

    # ## ===================================================================

    # ## The following features are features directly called by our model

    # The list of all the raw features
    se_full_raw_features = set(
        ls_demographic_features
        + ls_body_features
        + ls_medical_features
        + ls_health_status_features
        + ls_family_history_features
        + ls_health_behavior_features
        + ls_social_features
        + ls_healthcare_features
        + ls_ada_score
        + ls_lab_features
    )

    # The list of all the features without the lab features
    se_wolab_raw_features = se_full_raw_features - set(ls_lab_features)

    # The list of all the raw features and the encoded features
    se_full_encoded_features = se_full_raw_features | set(ls_encoded_features)

    # The list of all the features without the lab features and the encoded features
    se_wolab_encoded_features = se_wolab_raw_features | set(ls_encoded_features)

    se_iu_exclude_features = set(
        [
            # diet score
            "d_totaldietadjs",
            # Physical activity
            "phy_score",
            "phy_score_raw",
            "phy_score_0",
            "phy_score_1",
            "phy_score_2",
            "phy_score_255",
            "ada_physical",
            "a_score",
            # sleep time
            "s_sleep",
            # education
            "s_educ",
            "s_educ_0",
            "s_educ_1",
            "s_educ_2",
            "s_educ_3",
            # poverty
            "s_poverty",
            "s_poverty_0",
            "s_poverty_1",
            "s_poverty_2",
            "s_poverty_3",
        ]
    )

    # # TODO: to remove the se_excluded_features and the relative operations when the data is fixed
    # # Utility list of
    # # the features to be excluded because one version of our data deleted / renamed the original features when peforming one-hot encoding
    # se_encoded_features = set(
    #     [
    #         "s_private",
    #         "s_alcohol",
    #         "s_medicaid",
    #         "gestational_diabetes",
    #         "s_female",
    #         "s_race",
    #         "s_educ",
    #         "s_smoke4",
    #         "phy_score",
    #         "s_poverty",
    #         "s_access",
    #         "s_married",
    #     ]
    # )
    # se_wolab_encoded_features = se_wolab_encoded_features - se_encoded_features
    # # TODO: the above part is to be removed

    # ## ===================================================================
    # ## The following dictionary is used to map the raw features to the covariance to the label in order to rank the features

    dc_encode_map_raw = {
        # "id_subj": [],
        # "id_dsrvyr": [],
        # "id_mvpsu": [],
        # "id_mvstra": [],
        "s_insurance": [],
        "s_insgov": [],
        "s_smoke": [],
        "s_sleep": [],
        "s_age": ["s_age_raw", "ada_age"],
        "l_bmxwt": [],
        "l_bmxbmi": ["l_bmxbmi_raw", "ada_bmi"],
        "l_lbxtc": [],
        "l_lbxtr": [],
        "l_lbxgh": [],
        "l_bmxwaist": [],
        "l_lbxglu": [],
        "l_meansbp": [],
        "l_meandbp": [],
        "l_ldlnew": [],
        "l_bpmed": ["l_bpmed_raw", "ada_bpmed"],
        "l_cholmed": [],
        "l_dmmed": [],
        "l_dminsulin": [],
        "l_dmoral": [],
        "l_dmrisk": [],
        "l_bu": [],
        "l_ua": [],
        "l_cr": [],
        "l_nasi": [],
        "l_ksi": [],
        "d_totaldietadjs": [],
        "a_score": [],
        "hx_htn": [],
        "deprecated_dx_diagyes": [],
        "deprecated_dx_diagnoand": [],
        "deprecated_dx_diagnoor": [],
        "deprecated_dx_poordiagno": [],
        "deprecated_dx_poor": [],
        "deprecated_dx_prediag": [],
        "dx_undxdm": [],
        "dx_predm": [],
        "dx_metabolic": [],
        # "_mi_id": [],
        # "_mi_miss": [],
        # "_mi_m": [],
        "cluster_v1": [],
        "cluster_v2": [],
        # "cnt_1": [],
        # "cnt_2": [],
        # "cnt_3": [],
        # "cnt_4": [],
        # "cnt_5": [],
        # "cnt_6": [],
        # "cnt_7": [],
        # "cnt_8": [],
        # "cnt_9": [],
        # "cnt_10": [],
        "diabetes_label": [],
        "phy_score_year": [],
        "familydm_meanSBP": [],
        "familydm_meanDBP": [],
        "familydm_BPMed": [],
        "familydm": ["familydm_raw", "ada_familydm"],
        "ada_overall": [],
        "s_female": ["s_female_0", "s_female_1", "s_female_raw", "ada_gender"],
        "s_poverty": ["s_poverty_0", "s_poverty_1", "s_poverty_2", "s_poverty_3"],
        "s_race": ["s_race_0", "s_race_1", "s_race_2", "s_race_3"],
        "s_educ": ["s_educ_0", "s_educ_1", "s_educ_2", "s_educ_3"],
        "s_married": ["s_married_0", "s_married_1", "s_married_2"],
        "s_private": ["s_private_0", "s_private_1"],
        "s_medicaid": ["s_medicaid_0", "s_medicaid_1"],
        "s_access": ["s_access_0", "s_access_1", "s_access_2"],
        "s_smoke4": ["s_smoke4_0", "s_smoke4_1", "s_smoke4_2", "s_smoke4_3"],
        "s_alcohol": ["s_alcohol_0", "s_alcohol_1", "s_alcohol_2"],
        "phy_score": [
            "phy_score_0",
            "phy_score_1",
            "phy_score_2",
            "phy_score_255",
            "phy_score_raw",
            "ada_physical",
        ],
        "gestational_diabetes": [
            "gestational_diabetes_0",
            "gestational_diabetes_1",
            "gestational_diabetes_255",
            "gestational_diabetes_raw",
        ],
    }

    dc_encode_map = {
        "s_insurance": [],
        "s_insgov": [],
        "s_smoke": [],
        "s_sleep": [],
        "s_age": ["s_age_raw", "ada_age"],
        "l_bmxwt": [],
        "l_bmxbmi": ["l_bmxbmi_raw", "ada_bmi"],
        "l_lbxtc": [],
        "l_lbxtr": [],
        "l_lbxgh": [],
        "l_bmxwaist": [],
        "l_lbxglu": [],
        "l_meansbp": [],
        "l_meandbp": [],
        "l_ldlnew": [],
        "l_bpmed": ["l_bpmed_raw", "ada_bpmed"],
        "l_cholmed": [],
        "l_dmmed": [],
        "l_dminsulin": [],
        "l_dmoral": [],
        "l_dmrisk": [],
        "l_bu": [],
        "l_ua": [],
        "l_cr": [],
        "l_nasi": [],
        "l_ksi": [],
        "d_totaldietadjs": [],
        "a_score": [],
        "hx_htn": [],
        "deprecated_dx_diagyes": [],
        "deprecated_dx_diagnoand": [],
        "deprecated_dx_diagnoor": [],
        "deprecated_dx_poordiagno": [],
        "deprecated_dx_poor": [],
        "deprecated_dx_prediag": [],
        "dx_undxdm": [],
        "dx_predm": [],
        "dx_metabolic": [],
        "cluster_v1": [],
        "cluster_v2": [],
        "diabetes_label": [],
        "phy_score_year": [],
        "familydm_meanSBP": [],
        "familydm_meanDBP": [],
        "familydm_BPMed": [],
        "familydm": ["familydm_raw", "ada_familydm"],
        "ada_overall": [],
        "s_female": ["s_female_0", "s_female_1", "s_female_raw", "ada_gender"],
        "s_poverty": ["s_poverty_0", "s_poverty_1", "s_poverty_2", "s_poverty_3"],
        "s_race": ["s_race_0", "s_race_1", "s_race_2", "s_race_3"],
        "s_educ": ["s_educ_0", "s_educ_1", "s_educ_2", "s_educ_3"],
        "s_married": ["s_married_0", "s_married_1", "s_married_2"],
        "s_private": ["s_private_0", "s_private_1"],
        "s_medicaid": ["s_medicaid_0", "s_medicaid_1"],
        "s_access": ["s_access_0", "s_access_1", "s_access_2"],
        "s_smoke4": ["s_smoke4_0", "s_smoke4_1", "s_smoke4_2", "s_smoke4_3"],
        "s_alcohol": ["s_alcohol_0", "s_alcohol_1", "s_alcohol_2"],
        "phy_score": [
            "phy_score_0",
            "phy_score_1",
            "phy_score_2",
            "phy_score_255",
            "phy_score_raw",
            "ada_physical",
        ],
        "gestational_diabetes": [
            "gestational_diabetes_0",
            "gestational_diabetes_1",
            "gestational_diabetes_255",
            "gestational_diabetes_raw",
        ],
    }

    ls_cov_rank_raw = {
        # "id_subj": 1787.8891380828945,
        # "_mi_id": 678.2524727673563,
        # "cnt_4": -444.3171645928432,
        # "cnt_10": 435.66041306705176,
        # "cnt_7": 372.17665244195916,
        # "cnt_5": 352.63468514924904,
        # "cnt_2": 307.3903163887272,
        # "cnt_9": 258.1936655823665,
        # "cnt_8": -140.14081740109177,
        # "cnt_3": -135.08706698866132,
        # "cnt_6": -117.99028368904924,
        # "cnt_1": -79.99773637919658,
        "s_age_raw": 2.67734043680532,
        "l_lbxtc": 2.611777974709874,
        # "id_mvstra": 2.57643983879292,
        "l_bmxwaist": 2.3244677030097094,
        "gestational_diabetes": 2.20894284015186,
        "gestational_diabetes_raw": 2.20894284015186,
        "l_meansbp": 1.954359087489775,
        "l_bmxbmi_raw": 0.6326823402414672,
        "l_meandbp": 0.6005238249033721,
        "phy_score_year": 0.3430131912442474,
        "ada_overall": 0.34229728563199713,
        "l_nasi": 0.3419588028258517,
        "diabetes_label": 0.28434579338837945,
        "l_bu": 0.2501147067386505,
        "l_lbxgh": 0.18765655393777567,
        "ada_age": 0.17841984545073544,
        "id_dsrvyr": 0.1715065956221237,
        "dx_predm": 0.16288806309478476,
        "l_ua": 0.12898749983269514,
        "l_lbxtr": 0.11722798667184968,
        "l_lbxglu": 0.09840143558755393,
        "ada_bmi": 0.08481486124862199,
        "dx_metabolic": 0.0832711490971488,
        "cluster_v2": -0.08007885988513216,
        "s_smoke": -0.08001996853807125,
        "l_bpmed_raw": -0.07042513633973108,
        "familydm_BPMed": -0.07042513633973108,
        "l_bpmed": -0.07042513633973108,
        "phy_score_raw": -0.06781095484656988,
        "phy_score": -0.06781095484656988,
        "dx_undxdm": 0.06072886514679671,
        "deprecated_dx_diagnoor": 0.06072886514679671,
        "l_cholmed": -0.057733915976193546,
        "cluster_v1": -0.054610783575404484,
        "s_female_raw": -0.051402555879550754,
        "s_female": -0.051402555879550754,
        "l_dmrisk": -0.046584406244201716,
        "l_ldlnew": 0.04226132338764867,
        "s_smoke4": 0.04213675882212895,
        "s_age": 0.03996030502694511,
        "hx_htn": 0.03879426966072755,
        "deprecated_dx_poordiagno": 0.03608526769592266,
        "deprecated_dx_poor": 0.03608526769592266,
        "familydm_raw": -0.03412388868061904,
        "familydm": -0.03412388868061904,
        "s_educ": -0.03267294636380937,
        "ada_bpmed": 0.03199123850000947,
        "l_bmxwt": 0.030367293000421416,
        "ada_physical": 0.027029777582018997,
        "a_score": -0.025999449152245765,
        "s_alcohol": -0.024098582346766154,
        "l_bmxbmi": 0.0228068827951677,
        "ada_familydm": 0.020041562850611302,
        "s_race": 0.01840314074086436,
        "s_married": 0.017197218845083625,
        "l_ksi": 0.016901554567020253,
        "l_dmoral": -0.015560282526202984,
        "deprecated_dx_diagnoand": 0.01364199144601947,
        "s_private": 0.012830614561168057,
        "d_totaldietadjs": -0.011047383098649924,
        "s_sleep": -0.01056032634621066,
        "familydm_meanSBP": 0.010241428035823905,
        "s_insgov": -0.00785159429657696,
        "l_dmmed": -0.005614668611538266,
        "l_dminsulin": -0.005614668611538266,
        # "_mi_miss": -0.005511473682284622,
        "l_cr": 0.005352862724354965,
        "s_access": 0.005189327208109333,
        "familydm_meanDBP": 0.0035782600073301176,
        # "id_mvpsu": 0.0035052507972003805,
        "s_insurance": 0.002276636822735706,
        "s_poverty": -0.0015816919130365612,
        "s_medicaid": -0.0007025089359269314,
        # "_mi_m": -1.783596141385875e-18,
        # "s_poverty_0": 0.0,
        # "phy_score_255": 0.0,
        # "s_alcohol_0": 0.0,
        # "s_alcohol_1": 0.0,
        # "s_alcohol_2": 0.0,
        # "phy_score_0": 0.0,
        # "phy_score_1": 0.0,
        # "phy_score_2": 0.0,
        # "s_female_1": 0.0,
        # "gestational_diabetes_0": 0.0,
        # "gestational_diabetes_1": 0.0,
        # "gestational_diabetes_255": 0.0,
        # "s_poverty_1": 0.0,
        # "ada_gender": 0.0,
        # # "deprecated_dx_prediag": 0.0,
        # # "deprecated_dx_diagyes": 0.0,
        # "s_smoke4_3": 0.0,
        # "s_smoke4_2": 0.0,
        # "s_smoke4_1": 0.0,
        # "s_educ_3": 0.0,
        # "s_poverty_2": 0.0,
        # "s_poverty_3": 0.0,
        # "s_race_0": 0.0,
        # "s_race_1": 0.0,
        # "s_race_2": 0.0,
        # "s_race_3": 0.0,
        # "s_educ_0": 0.0,
        # "s_educ_1": 0.0,
        # "s_educ_2": 0.0,
        # "s_married_0": 0.0,
        # "s_smoke4_0": 0.0,
        # "s_married_1": 0.0,
        # "s_married_2": 0.0,
        # "s_private_0": 0.0,
        # "s_private_1": 0.0,
        # "s_medicaid_0": 0.0,
        # "s_medicaid_1": 0.0,
        # "s_access_0": 0.0,
        # "s_access_1": 0.0,
        # "s_access_2": 0.0,
        # "s_female_0": 0.0,
    }

    ls_cov_rank = {
        "s_age_raw": 2.67734043680532,
        "l_lbxtc": 2.611777974709874,
        "l_bmxwaist": 2.3244677030097094,
        "gestational_diabetes": 2.20894284015186,
        "gestational_diabetes_raw": 2.20894284015186,
        "l_meansbp": 1.954359087489775,
        "l_bmxbmi_raw": 0.6326823402414672,
        "l_meandbp": 0.6005238249033721,
        "phy_score_year": 0.3430131912442474,
        "ada_overall": 0.34229728563199713,
        "l_nasi": 0.3419588028258517,
        "diabetes_label": 0.28434579338837945,
        "l_bu": 0.2501147067386505,
        "l_lbxgh": 0.18765655393777567,
        "ada_age": 0.17841984545073544,
        "id_dsrvyr": 0.1715065956221237,
        "dx_predm": 0.16288806309478476,
        "l_ua": 0.12898749983269514,
        "l_lbxtr": 0.11722798667184968,
        # ## covariance > 0.1
        "l_lbxglu": 0.09840143558755393,
        "ada_bmi": 0.08481486124862199,
        "dx_metabolic": 0.0832711490971488,
        "cluster_v2": -0.08007885988513216,
        "s_smoke": -0.08001996853807125,
        "l_bpmed_raw": -0.07042513633973108,
        "familydm_BPMed": -0.07042513633973108,
        "l_bpmed": -0.07042513633973108,
        "phy_score_raw": -0.06781095484656988,
        "phy_score": -0.06781095484656988,
        "dx_undxdm": 0.06072886514679671,
        # "deprecated_dx_diagnoor": 0.06072886514679671,
        "l_cholmed": -0.057733915976193546,
        "cluster_v1": -0.054610783575404484,
        "s_female_raw": -0.051402555879550754,
        "s_female": -0.051402555879550754,
        # ## covariance > 0.05
        "l_dmrisk": -0.046584406244201716,
        "l_ldlnew": 0.04226132338764867,
        "s_smoke4": 0.04213675882212895,
        "s_age": 0.03996030502694511,
        "hx_htn": 0.03879426966072755,
        # "deprecated_dx_poordiagno": 0.03608526769592266,
        # "deprecated_dx_poor": 0.03608526769592266,
        "familydm_raw": -0.03412388868061904,
        "familydm": -0.03412388868061904,
        "s_educ": -0.03267294636380937,
        "ada_bpmed": 0.03199123850000947,
        "l_bmxwt": 0.030367293000421416,
        "ada_physical": 0.027029777582018997,
        "a_score": -0.025999449152245765,
        "s_alcohol": -0.024098582346766154,
        "l_bmxbmi": 0.0228068827951677,
        "ada_familydm": 0.020041562850611302,
        # ## covariance > 0.02
        "s_race": 0.01840314074086436,
        "s_married": 0.017197218845083625,
        "l_ksi": 0.016901554567020253,
        "l_dmoral": -0.015560282526202984,
        # "deprecated_dx_diagnoand": 0.01364199144601947,
        "s_private": 0.012830614561168057,
        "d_totaldietadjs": -0.011047383098649924,
        "s_sleep": -0.01056032634621066,
        "familydm_meanSBP": 0.010241428035823905,
        # ## covariance > 0.01
        "s_insgov": -0.00785159429657696,
        "l_dmmed": -0.005614668611538266,
        "l_dminsulin": -0.005614668611538266,
        "l_cr": 0.005352862724354965,
        "s_access": 0.005189327208109333,
        "familydm_meanDBP": 0.0035782600073301176,
        "s_insurance": 0.002276636822735706,
        "s_poverty": -0.0015816919130365612,
        "s_medicaid": -0.0007025089359269314,
    }

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        print(f"===== Data Loaded: {file_path} =====")

    def _select_features_cov(self, cov_idx, mode="wolab"):
        """Extracts features from self.ls_cov_rank with absolute covariance > cov_idx"""
        if mode == "wolab":
            cov_features = [
                feature
                for feature, value in self.ls_cov_rank.items()
                if abs(value) > cov_idx and feature not in self.ls_lab_features
            ]
        elif mode == "full":
            cov_features = [
                feature
                for feature, value in self.ls_cov_rank.items()
                if abs(value) > cov_idx
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        selected_features = []

        for feature in cov_features:
            if feature in self.dc_encode_map:
                selected_features.append(feature)
                selected_features.extend(self.dc_encode_map[feature])

            for key, val in self.dc_encode_map.items():
                if feature in val:
                    selected_features.append(key)

        cleaned_list = list(set(selected_features) - self.se_excluded_features)

        cleaned_list = [i for i in cleaned_list if i in self.ls_full_features]

        cleaned_list = sorted(cleaned_list)

        # Returns list of features for self.classify_features
        return cleaned_list

    def data_process_cluster(
        self,
        classify_input=None,
        cluster_method=None,
        cluster_idx=None,
    ):
        """
        cluster_method: None, "cluster_v1", "cluster_v2". None means all clusters combined.
        cluster_idx: 0, 1, 2, None. None means all clusters combined.
        """

        # # ## Define the features to be used for clustering and classification
        # self.cluster_features = (
        #     self.ls_categorical_features + self.ls_numerical_features
        # )

        print(f"   -- classify_input: {classify_input}")
        if classify_input == "full_raw" or classify_input == "fullRaw":
            self.classify_features = sorted(list(self.se_full_raw_features))
        elif classify_input == "full_enc" or classify_input == "fullEnc":
            self.classify_features = sorted(list(self.se_full_encoded_features))
        elif classify_input == "wo_lab_raw" or classify_input == "woLabRaw":
            self.classify_features = sorted(list(self.se_wolab_raw_features))
        elif classify_input == "wo_lab_enc" or classify_input == "woLabEnc":
            self.classify_features = sorted(list(self.se_wolab_encoded_features))
        elif classify_input == "iu_raw" or classify_input == "iuRaw":
            self.classify_features = sorted(
                list(self.se_wolab_raw_features - self.se_iu_exclude_features)
            )
        elif classify_input == "iu_enc" or classify_input == "iuEnc":
            self.classify_features = sorted(
                list(self.se_wolab_encoded_features - self.se_iu_exclude_features)
            )

        # elif classify_input.startswith("cov>"):
        #     cov_idx = float(classify_input.split(">")[1])
        #     self.classify_features = self._select_features_cov(
        #         cov_idx
        #     )  # Selects features with cov > xxx
        #     print(f"   -- classify_input: {classify_input}")
        #     print(f"   -- cov_idx: {cov_idx}")
        #     print(f"   -- len(self.classify_features): {len(self.classify_features)}")
        #     print(f"   -- self.classify_features: {self.classify_features}")
        # elif classify_input.startswith("fullcov>"):
        #     cov_idx = float(classify_input.split(">")[1])
        #     self.classify_features = self._select_features_cov(
        #         cov_idx,
        #         mode="full",
        #     )  # Selects features with cov > xxx
        #     print(f"   -- classify_input: {classify_input}")
        #     print(f"   -- cov_idx: {cov_idx}")
        #     print(f"   -- len(self.classify_features): {len(self.classify_features)}")
        #     print(f"   -- self.classify_features: {self.classify_features}")
        # elif classify_input == "wo_lab_diet":
        #     self.classify_features = list(
        #         set(self.ls_wolab_features) - set(["d_totaldietadjs"])
        #     )
        # elif classify_input == "all":
        #     self.classify_features = (
        #         self.ls_categorical_features
        #         + self.ls_numerical_features
        #         + self.ls_lab_features
        #     )
        # elif classify_input == "without_lab":
        #     self.classify_features = (
        #         self.ls_categorical_features + self.ls_numerical_features + ["l_bpmed"]
        #     )
        # elif classify_input == "without_lab_encoded":
        #     self.classify_features = list(
        #         set(
        #             self.ls_numerical_features
        #             + self.ls_encoded_features
        #             + ["l_bpmed"]
        #             + self.ls_ada_score
        #         )
        #     )
        # elif classify_input == "no_lab_strict":
        #     self.classify_features = (
        #         self.ls_categorical_features + self.ls_numerical_features
        #     )
        # elif classify_input == "no_lab_strict_encoded":
        #     self.classify_features = list(
        #         set(
        #             self.ls_numerical_features
        #             + self.ls_encoded_features
        #             + self.ls_ada_score
        #         )
        #     )
        # elif classify_input == "priviledged_info_encoded":
        #     self.classify_features = list(
        #         set(
        #             self.ls_numerical_features
        #             + self.ls_encoded_features
        #             + self.ls_ada_score
        #             + ["l_lbxglu", "l_ldlnew", "l_lbxtr", "l_ua", "l_bu"]
        #         )
        #     )
        # elif classify_input == "ada":
            self.classify_features = self.ls_ada_self_test_features
        
        else:
            raise ValueError(f"Unknown classify_input: {classify_input}")

        # NOTE: This step is very important to make sure the order of the features is consistent
        self.classify_features.sort()

        df = self.df
        if cluster_idx is not None and cluster_method is not None:
            df = df[df[cluster_method] == cluster_idx].copy()
            print(f"   -- cluster_idx: {cluster_idx}; cluster_method: {cluster_method}")
        else:
            df = df.copy()
            print(f"   -- Use all clusters combined")
        df = df[self.classify_features + ["diabetes_label"]].copy()

        # NOTE: 0: Healthy (no diabetes); 1: diabetes + pre-diabetes
        df["diabetes_label"] = df["diabetes_label"].apply(lambda x: 0 if x == 0 else 1)

        print(f"   -- classify_input: {classify_input}")
        print(f"   -- classify_features:\n{self.classify_features}")

        print("===== Data Pre-processing Complete! =====")

        return df
