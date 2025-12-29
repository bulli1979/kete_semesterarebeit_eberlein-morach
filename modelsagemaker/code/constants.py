########################################################################################################################
#                                                ENVIRONMENT VARIABLES                                                 #
########################################################################################################################
NUM_GPU = "NUM_GPU"
SAGEMAKER_INFERENCE_OUTPUT = "SAGEMAKER_INFERENCE_OUTPUT"

########################################################################################################################
#                                                OUTPUT CONSTANTS                                                      #
########################################################################################################################
PROBABILITY = "probability"
PROBABILITIES = "probabilities"
PREDICTED_LABEL = "predicted_label"
LABELS = "labels"
PREDICTIONS = "predictions"

########################################################################################################################
#                                                DATA FORMAT CONSTANTS                                                 #
########################################################################################################################
JSON_FORMAT = "application/json"
JSON_LINES_FORMAT = "application/jsonlines"
CSV_FORMAT = "text/csv"
COMMA_DELIMITER = ","
BRACKET_FORMATTER = "{}"
NEW_LINE_CHARACTER = "\n"
ALLOWED_INPUT_FORMATS = [CSV_FORMAT]
ALLOWED_OUTPUT_FORMATS = [JSON_FORMAT, CSV_FORMAT, JSON_LINES_FORMAT]

########################################################################################################################
#                                                INFERENCE DATA CONSTANTS                                              #
########################################################################################################################
COLUMN_NAMES = "column_names"
