import datetime

labeled = "data/labeledTrainData.tsv"
unlabeled = "data/unlabeledTrainData.tsv"
test_data = "data/testData.tsv"

output_template = "output/%s-forest-bag-words.csv"

def get_output_name():
    current = datetime.datetime.now()
    return(output_template % current.strftime("%Y.%m.%d.%H.%M.%S"))
