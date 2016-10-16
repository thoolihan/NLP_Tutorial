# constants for project
import datetime

labeled = "data/labeledTrainData.tsv"
unlabeled = "data/unlabeledTrainData.tsv"
test_data = "data/testData.tsv"

w2v_model = "300features_40minwords_10context"
num_features = 300

output_template = "output/%s-%s.csv"

def get_output_name(model='forest-bag-words'):
    current = datetime.datetime.now()
    return(output_template % (current.strftime("%Y.%m.%d.%H.%M.%S"), model))
