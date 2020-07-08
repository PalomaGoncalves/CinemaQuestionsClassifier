from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import sys

moviesNames = open('recursos/list_movies.txt').read().replace('\t', '').split('\n')

tagsAndQuestion = {
    "actor_name": [],
    "budget": [],
    "character_name": [],
    "genre": [],
    "keyword": [],
    "original_language": [],
    "original_title": [],
    "overview": [],
    "person_name": [],
    "production_company": [],
    "production_country": [],
    "release_date": [],
    "revenue": [],
    "runtime": [],
    "spoken_language": [],
    "vote_avg": []
}


def openFile(file):
    new=[]
    text = open(file, encoding="utf8").read().replace('\t', '').split('\n')
    for i in text:
        new.append(replaceByMovie(i))
    return new


def replaceByMovie(string):
    for movie in moviesNames:
        if len(movie)>1 and (string.find(movie))!=-1:
            string = string.replace(movie, "moviesname")
            break
    return string


def separateTagsFromQuestions(f):
    array=openFile(f)
    for i in array:
        aux = i.split(' ', 1)
        if(len(aux) > 1):
            tagsAndQuestion[aux[0]].append(aux[1])
    return tagsAndQuestion

def accuracy(array1, array2):
    tp = 0
    for i in range(len(array1) - 1):
        if array1[i] == array2[i]:
            tp += 1
        else:
            print(str(i + 1) + " " + array1[i] + " " + array2[i])
    print(tp)
    return (tp / len(array1))


def make_document(array):
    
    document = ''
    for string in array:
        document += string
    return [document]


def similarity_without_stop_words(string, questions):   
    stopWords = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stopWords)
    questionsFrequency = vectorizer.fit_transform(questions)
    termFrequency = vectorizer.transform([string])
    cosine = cosine_similarity(questionsFrequency, termFrequency, dense_output=True)
    return cosine[0][0]


def calculateSimilarity(q2):
    results = []
    for new in q2:
        anwsers = {
        "actor_name": 0,
        "budget": 0,
        "character_name": 0,
        "genre": 0,
        "keyword": 0,
        "original_language": 0,
        "original_title": 0,
        "overview": 0,
        "person_name": 0,
        "production_company": 0,
        "production_country": 0,
        "release_date": 0,
        "revenue": 0,
        "runtime": 0,
        "spoken_language": 0,
        "vote_avg": 0
    }
        
        for k, v in tagsAndQuestion.items():
            anwsers[k] = jacard(new, tagsAndQuestion[k])
        aux = ['', 0]
        for x, y in anwsers.items():
            if y > aux[1]:
                aux[1] = y
                aux[0] = x
        print(aux[0])
        results.append(aux[0])
    new_results = openFile("corpora/NovasQuestoesResultados.txt")
    for i in range(len(new_results)):
        new_results[i] = new_results[i].replace(" ", "")
    #print(accuracy(results, new_results))


def jacard(str1, array):
    max = 0
    avg = 0
    for str2 in array:
        stopWords = stopwords.words('english')
        countVectorizer = CountVectorizer(stop_words=stopWords)
        vocabulario = countVectorizer.fit_transform([str1 + str2])
        str1frequency = countVectorizer.transform([str1])
        str2frequency = countVectorizer.transform([str2])
        intersection = 0
        for x in range(len(str1frequency.toarray()[0])):
            if str1frequency.toarray()[0][x] > 0 and str2frequency.toarray()[0][x] > 0: 
                intersection += 1
        if (intersection / len(str1frequency.toarray()[0])) > max:
            max = intersection / len(str1frequency.toarray()[0])
        avg += intersection / len(str1frequency.toarray()[0])
    return max


if __name__ == '__main__':

    arg1=str(sys.argv[1])
    arg2=str(sys.argv[2])
    separateTagsFromQuestions(arg1)  
    calculateSimilarity(openFile(arg2))
