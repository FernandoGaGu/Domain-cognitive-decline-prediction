library(lavaan)
library(arrow)

# Windows setup
setwd('')

# ============= Input parameters
# files generated using "notebook/ADNI-Composites-calculation.ipynb"
data.file.no.missing <- "../../data/adni/processed/20240428_neuropsycho_inputSEM_allObs.parquet"

# files generated using "notebook/ADNI-Composites-calculation.ipynb"
data.file.no.missing.last <- "../../data/adni/processed/20240428_neuropsycho_inputSEM_last.parquet"

# variables used to calculate the different composites
facVars <- c(
  'memory_avlt_trial_1',
  'memory_avlt_trial_2',
  'memory_avlt_trial_6',
  'memory_avlt_delayed',
  'memory_avlt_recognition',
  'memory_word_recognition',
  'memory_word_recall_delayed',
  'memory_word_recall',
  
  'language_bnt_tot',
  'language_cat_fluency',
  'language_naming',
  'language_word_finding_diff',
  
  'exec_tmt_a_time',
  'exec_tmt_b_time',
  'attention_number_cancellation',
  
  'visuos_clock_copy_tot_score',
  'visuos_clock_draw_tot_score',
  'visuos_constructional_praxis',
  'visuos_ideational_praxis'
)


# load the data 
data.last <- read_parquet(data.file.no.missing.last)
data <- read_parquet(data.file.no.missing)


# definition of the factorial structure to be estimated (releasing the first 
# factor loadings, and identifying the model by setting the variances of the 
# latent factors to 1). The different cognitive functions have been considered 
# as independent factors (as is done in the methodology proposed in ADNI).

##############################################################################
########################### MEMORY COMPOSITE #################################
##############################################################################

memory.sem.definition <- '
  # measurement model
  memory =~ NA*memory_avlt_trial_6 + memory_avlt_trial_1 + memory_avlt_trial_2 + 
  memory_avlt_delayed + memory_avlt_recognition +
  memory_word_recognition + memory_word_recall_delayed +
  memory_word_recall

  # fix factor loading variances to 1
  memory ~~ 1*memory
'

# fit the model
memory.sem.model <- cfa(
  memory.sem.definition, data=data.last, std.lv=FALSE, estimator='MLR')
summary(memory.sem.model, fit.measures=TRUE, standardized=TRUE)

# estimate the factor score for all the observations
memory.facScores.no.missing <- predict(memory.sem.model, newdata=data)   

# add subject level information
memory.facScores.no.missing <- data.frame(memory.facScores.no.missing)
memory.facScores.no.missing$subject_id = data$subject_id
memory.facScores.no.missing$neurobat_date = data$neurobat_date



##############################################################################
####################### ATTENTION & EXEC COMPOSITE ###########################
##############################################################################

exec.sem.definition <- '
  # measurement model
  exec =~ NA*exec_tmt_b_time + exec_tmt_a_time + attention_number_cancellation
  
  # fix factor loading variances to 1
  exec ~~ 1*exec
'

# adjust the model
exec.sem.model <- cfa(
  exec.sem.definition, data=data.last, std.lv=FALSE, estimator='WLSMV',
  ordered = c(
    "attention_number_cancellation"
    ))
summary(exec.sem.model, fit.measures=TRUE, standardized=TRUE)


# estimate the factor score for all the observations
exec.facScores.no.missing <- predict(exec.sem.model, newdata=data)   

# add subject level information
exec.facScores.no.missing <- data.frame(exec.facScores.no.missing)
exec.facScores.no.missing$subject_id = data$subject_id
exec.facScores.no.missing$neurobat_date = data$neurobat_date


##############################################################################
############################ LANGUAGE COMPOSITE ##############################
##############################################################################

language.sem.definition <- '
  # measurement model
  language =~ NA*language_bnt_tot + language_cat_fluency + language_naming + language_word_finding_diff

  # fix factor loading variances to 1
  language ~~ 1*language
'

# fit the model
language.sem.model <- cfa(
    language.sem.definition, data=data.last, std.lv=FALSE, estimator='WLSMV',
    ordered = c("language_naming", "language_word_finding_diff")
    )
summary(language.sem.model, fit.measures=TRUE, standardized=TRUE)

# estimate the factor score for all the observations
language.facScores.no.missing <- predict(language.sem.model, newdata=data)   

# add subject level information
language.facScores.no.missing <- data.frame(language.facScores.no.missing)
language.facScores.no.missing$subject_id = data$subject_id
language.facScores.no.missing$neurobat_date = data$neurobat_date


##############################################################################
########################## VISUOSPATIAL COMPOSITE ############################
##############################################################################

visuospatial.sem.definition <- '
  # measurement model
  visuospatial =~ NA*visuos_clock_draw_tot_score + visuos_clock_copy_tot_score + 
  visuos_constructional_praxis + visuos_ideational_praxis

  # fix factor loading variances to 1
  visuospatial ~~ 1*visuospatial
'

# fit the model
visuospatial.sem.model <- cfa(
  visuospatial.sem.definition, data=data.last, std.lv=TRUE, estimator='WLSMV',
  ordered = c(
    "visuos_clock_draw_tot_score", 
    "visuos_clock_copy_tot_score",
    "visuos_constructional_praxis", 
    "visuos_ideational_praxis")
)
summary(visuospatial.sem.model, fit.measures=TRUE, standardized=TRUE)

# estimate the factor score for all the observations
visuospatial.facScores.no.missing <- predict(visuospatial.sem.model, newdata=data)   

# add subject level information
visuospatial.facScores.no.missing <- data.frame(visuospatial.facScores.no.missing)
visuospatial.facScores.no.missing$subject_id = data$subject_id
visuospatial.facScores.no.missing$neurobat_date = data$neurobat_date


################# MERGE INFORMATION

# concatenate all the dataframes
final.facScores.no.missing <- merge(
  memory.facScores.no.missing, 
  exec.facScores.no.missing, 
  by=c("subject_id", "neurobat_date"), all=TRUE)

final.facScores.no.missing <- merge(
  final.facScores.no.missing, 
  language.facScores.no.missing, 
  by=c("subject_id", "neurobat_date"), all=TRUE)

final.facScores.no.missing <- merge(
  final.facScores.no.missing, 
  visuospatial.facScores.no.missing, 
  by=c("subject_id", "neurobat_date"), all=TRUE)


# export the generated data
out.file.name <- paste("../../data/adni/processed", "20240428_neuropsycho_lavaan_SEM.parquet", sep='/')
write_parquet(final.facScores.no.missing, out.file.name)



