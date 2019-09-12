import utils

if utils.query_yes_no("Do you want to clean all previously trained models ?", default="no"):
    utils.cleanModels()
    print("DONE")
else:
    print("Cancelling")
