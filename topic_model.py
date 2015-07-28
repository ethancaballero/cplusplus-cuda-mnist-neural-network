#source todo-api/flask/bin/activate & source dato-env/bin/activate
#source myproject/venv/bin/activate   <---for flask on hive centos
#pip install flask
#^^^ORDER MATTERS FOR LINE ABOVE

#http://flask.pocoo.org/docs/0.10/patterns/fileuploads/
#https://dato.com/products/create/docs/generated/graphlab.topic_model.create.html
#source dato-env/bin/activate
#python
import graphlab as gl




docs = graphlab.SArray('posts_100K.tsv')

m = graphlab.topic_model.create(docs)

m2 = graphlab.topic_model.create(docs, initial_topics=m['topics'])

from graphlab import SFrame

associations = SFrame({'word':['hurricane', 'wind', 'storm'],
                           'topic': [0, 0, 0]})
m = graphlab.topic_model.create(docs,
                                    associations=associations
                                    num_topics=20,       # number of topics
                                    num_iterations=10,   # algorithm parameters
                                    alpha=.01, beta=.1)  # hyperparameters)


m.list_fields()




'''
#sorted model
model = original_model['cluster_info'].sort('size', ascending = False)

wo = model['Violation Code']
xo = model['Violation Location']
yo = model['size']
zo = model[['Violation Code','Violation Location']]

#Should I be using sum_squared_distance for anything?
#Correlation such as the size/number of Violation Codes when the violation occurs in a certain Location

#yo = model['cluster_info']['size']  #<--old version

w=list(wo)
x=list(xo)
y=list(yo)
z=list(zo)

s = 'The violations codes that occured most (in descending order):' + repr(w)
print s

t = 'The locations where violations occured most (in descending order):' + repr(x)
print t

u = 'The violation_codes:violation_location combos that were most correlated (in descending order):' + repr(z)
print u

import matplotlib.pyplot as plt
figure,axis = plt.subplots()
axis.set_title('Violation Code : Size')
plt.bar(w, y)
plt.show()

figure,axis = plt.subplots()
axis.set_title('Violation Locations: Size')
plt.bar(x, y)
plt.show()

figure,axis = plt.subplots()
axis.set_title('Amount of Violation Codes at Violation Locations: Size')
plt.bar(z, y)
plt.show()




#---old version
ex=model['num_examples']
num=model['num_clusters']

cluster=y*randn(1,ex/num))

import matplotlib.pyplot as plt
figure,axis = plt.subplots(1,1)
axis.set_title('Violation Code : Size')
plt.plot(w, y, marker='o', linestyle='  ')
plt.show

figure,axis = plt.subplots(1,1)
axis.set_title('Violation Location : Size')
plt.plot(x, y, marker='o', linestyle='  ')
plt.show()
#'''and None

#while(True):
if __name__ == '__main__':
    middle.run(port = 9000, debug=True)

#USER NEEDS TO RUN THE FOLLOWING COMMANDS IN PYTHON ONCE THIS PROGRAM IS DONE RUNNING:
#FOR UPLOADS RUN THIS:
#import requests
#files = {'file': ('parking_violations_short.csv', open('parking_violations_short.csv', 'rb'), 'text/csv', {'Expires': '0'})}
#r = requests.post('http://127.0.0.1:9000', files=files)
#r.text    #for test of what was uploaded

#FOR DOWNLOADS RUN THIS:
#import requests
#r = requests.get('http://127.0.0.1:9000/correlations')
#r.text