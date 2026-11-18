import _pickle as pickle

# my modules
from embed_new import embed
from visualize import plot_projection


def main():
    
    # unpickle complaints dictionary
    with open('data/many_complaints.bin', 'rb') as file:
        complaints_dict = pickle.load(file)
        
    # Generate a list of 50 customer complaints
    complaints = []
    true_labels = []

    for key, some_complaints in complaints_dict.items():
        for complaint in some_complaints:
                complaints.append(complaint)
                true_labels.append(key)
            
    # define transformer models to be used
    model_names = ["all-MiniLM-L6-v2", "gte-base-en-v15", "gte-large-en-v15", "mxbai-embed-large-v1", "GIST-small-Embedding-v0"]
    repos = ["all-MiniLM-L6-v2", "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5", "mixedbread-ai/mxbai-embed-large-v1", "avsolatorio/GIST-small-Embedding-v0"]
    repos_dict = {name: repo for name,repo in zip(model_names,repos)}
    # embed sentences
    transformers = {name: embed(repos_dict[name]) for name in model_names}
    #embeddings = {name: embed(complaints, repos_dict[name]) for name in model_names}
    embeddings = {name: transformers[name](complaints) for name in model_names}
    
    # visualize data with true labels
    for name in model_names:
        plot_name = name+'_ground_truth'
        print(plot_name)
        plot_projection(embeddings[name], true_labels, plot_name)
    
    
if __name__ == '__main__':
    main()
    