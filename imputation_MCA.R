library(foreign)
library(dplyr)
library(mltools)
library(data.table)
library(mice)
library(finalfit)
library(naniar)
library(GGally)
library(missMDA)
library(ggplot2)
library(tidyr)

data <- read.spss('data/DB_HER2low_20240312.sav', to.data.frame = TRUE, 
                  add.undeclared.levels = 'no', 
                  use.value.labels = F)

# Remove patient code (useless)
data <- data %>% select(-Pt_code)

# Some variables are unrelated like site of biopsy or timing of biopsy
site_timing_biopsy_var <- grep("site|timing", names(data), ignore.case = T)
data <- data %>% select(-all_of(site_timing_biopsy_var))

# Remove sex because there are too few males
data <- data %>% select(-Sex)

data_for_missing_plot <- data %>%
    rename('Neoadjuvant therapy: any' = Neoadj_therapy_any,
           'Neoadjuvant therapy: CT' = Neoadj_CT,
           'Primary BC histology' = PrimaryBC_histology,
           'Primary BC grade' = PrimaryBC_grade,
           'Primary BC HER2 phenotype category' = PrimaryBC_HER2_3cat,
           'Primary BC phenotype: TNBC' = PrimaryBC_Phenotype_TNBC,
           'Primary BC phenotype: ER low' = PrimaryBC_Phenotype_ER_low,
           'Primary BC phenotype: luminal' = PrimaryBC_Phenotype_Luminale,
           'Primary BC PgR (cut-off 14)' = PrimaryBC_PgR_14_cutoff,
           'Primary BC ki67 (cut-off 20)' = PrimaryBC_ki67_20_cutoff,
           'Stage 4 at diagnosis' = Stage4_at_diagnosis,
           'Adj CT' = Adj_CT,
           'Adj ET' = Adj_ET,
           'Recurrence: local' = Recurrence_local,
           'Recurrence: distant' = Recurrence_distant,
           'Distant recurrence site: visceral' = Distant_recurrence_visceral,
           'Distant recurrence site: lymphnodes' = Distant_recurrence_lymphnodes,
           'Distant recurrence site: soft/skin' = Distant_recurrence_softandskin,
           'Distant recurrence site: bone' = Distant_recurrence_bone,
           'Distant recurrence site: liver' = Distant_recurrence_liver,
           'Distant recurrence site: lung' = Distant_recurrence_lung,
           'Distant recurrence site: brain' = Distant_recurrence_brain,
           'Distant recurrence site: unusual' = Distant_recurrence_unusual,
           'Distant recurrence site: GI' = Distant_recurrence_GI,
           'Distant recurrence site: GU' = Distant_recurrence_GU,
           'Recurrence HER2 phenotype category' = Recurrence_phenotype_HER2_3cat,
           'Recurrence phenotype: TNBC' = Recurrence_phenotype_TNBC,
           'Recurrence phenotype: ER low' = Recurrence_phenotype_ERlow,
           'Recurrence phenotype: luminal' = Recurrence_phenotype_Luminale,
           'Age (cut-off 50 years)' = Age_50yrs_cutoff,
           'Lines therapies before biopsy (none or > 0)' = Lines_therapies_before_biopsy_none_versus_morethan0,
           'Lines therapies before biopsy CDK46i' = Lines_therapies_before_biopsy_CDK46i)

# Calculate the proportion of missing values for each variable
missing_data_prop <- data_for_missing_plot %>%
  summarise(across(everything(), ~mean(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Missing_Proportion")

missing_plot_custom <- missing_plot(data_for_missing_plot, use_labels = T, title = "") +
    theme_minimal() +
    scale_fill_gradient(low = "#56106e", high = "#fbbe22") +
    theme(legend.position = "none") +
    xlab("Patients") +
    xlim(0, 1300)

png('missingness_plot.png', width = 9000, height = 6000, res = 1200)
print(missing_plot_custom)
dev.off()

data_impute <- imputeMCA(data %>% mutate_all(factor), seed=101)
data <- data_impute$completeObs

# Building the target of interest: patients switching from negative to low HER2
data <- one_hot(as.data.table(data), cols = c("PrimaryBC_HER2_3cat", "Recurrence_phenotype_HER2_3cat"))
data$Switch_HER2_low_gain <- ifelse(data$PrimaryBC_HER2_3cat_0 == 1 & data$Recurrence_phenotype_HER2_3cat_1 == 1, 1, 0) %>% as.factor()

# Remove variables that determine the target
data <- data %>% select(-PrimaryBC_HER2_3cat_0, -PrimaryBC_HER2_3cat_1, -PrimaryBC_HER2_3cat_2, 
                        -Recurrence_phenotype_HER2_3cat_0, -Recurrence_phenotype_HER2_3cat_1, -Recurrence_phenotype_HER2_3cat_2)

# Some variables are clinically not meaningful because they are known after the biopsy
recurrence_pheno_var <- grep("Recurrence_phenotype", names(data), ignore.case = T)
data <- data %>% select(-all_of(recurrence_pheno_var))


# output data
write.csv2(data, 'data/cleaned_imputed_MCA_data_filtered.csv', quote = F, row.names = F)

