from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score

dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']



if __name__ == '__main__':
    agree = 0
    disagree = 0 
    discuss = 0
    unrelated = 0

    for stance in dev_data:
        if (stance['Stance'] == "agree"):
            agree+= 1
        elif (stance['Stance'] == "disagree"):
            disagree += 1
        elif (stance['Stance'] == "discuss"):
            discuss += 1
        else:
            unrelated += 1
    total = len(dev_data)
    print("Total examples: ", total)
    print("Agrees: ", agree / total)
    print("Disagrees: ", disagree / total)
    print("Discusses:" , discuss / total)
    print("Unrelated: ", unrelated / total)
'''
    for stance in training_data:
        print(stance)
        print(dataset.articles[stance['Body ID']])
        print("")


    for stance in dev_data:
        print(stance)
        print(dataset.articles[stance['Body ID']])
        print("")


    for stance in test_data:
        print(stance)
        print(dataset.articles[stance['Body ID']])
        print("")
'''