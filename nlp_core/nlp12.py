from nltk.corpus import wordnet

################# Synonyms And Antonyms #################

synonyms = []
antonyms = []

synsets = wordnet.synsets('good')
# print(synsets)

for syn in synsets:
    # print(syn)
    for s in syn.lemmas():
        # print(s.name())
        synonyms.append(s.name())
        for a in s.antonyms():
            antonyms.append(a.name())

# print(synonyms)
print(set(synonyms))
print(set(antonyms))
