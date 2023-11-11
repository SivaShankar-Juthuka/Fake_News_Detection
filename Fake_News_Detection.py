import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("C:/Users/Siva Shankar/Desktop/Fake News Detection/news.csv", error_bad_lines=False, engine='python')

x = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

tfvect = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)

classifier = PassiveAggressiveClassifier(max_iter = 50)
classifier.fit(tfid_x_train, y_train)

y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred) * 100
print('Accuracy :', score)

conf_mat = confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
print(conf_mat)

def fake_news_data(news):
  input_data = [news]
  vec_input_data = tfvect.transform(input_data)
  prediction = classifier.predict(vec_input_data)
  print(prediction)

import pickle
pickle.dump(classifier, open('model.pkl', 'wb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))

def fake_news_data(news):
  input_data = [news]
  vec_input_data = tfvect.transform(input_data)
  prediction = classifier.predict(vec_input_data)
  print(prediction)

fake_news_data("U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.Kerry said he expects to arrive in Paris Thursday evening, as he heads home after a week abroad. He said he will fly to France at the conclusion of a series of meetings scheduled for Thursday in Sofia, Bulgaria. He plans to meet the next day with Foreign Minister Laurent Fabius and President Francois Hollande, then return to Washington.The visit by Kerry, who has family and childhood ties to the country and speaks fluent French, could address some of the criticism that the United States snubbed France in its darkest hour in many years.The French press on Monday was filled with questions about why neither President Obama nor Kerry attended Sundayâ€™s march, as about 40 leaders of other nations did. Obama was said to have stayed away because his own security needs can be taxing on a country, and Kerry had prior commitments.Among roughly 40 leaders who did attend was Israeli Prime Minister Benjamin Netanyahu, no stranger to intense security, who marched beside Hollande through the city streets. The highest ranking U.S. officials attending the march were Jane Hartley, the ambassador to France, and Victoria Nuland, the assistant secretary of state for European affairs. Attorney General Eric H. Holder Jr. was in Paris for meetings with law enforcement officials but did not participate in the march.Kerry spent Sunday at a business summit hosted by Indiaâ€™s prime minister, Narendra Modi. The United States is eager for India to relax stringent laws that function as barriers to foreign investment and hopes Modiâ€™s government will act to open the huge Indian market for more American businesses.In a news conference, Kerry brushed aside criticism that the United States had not sent a more senior official to Paris as â€œquibbling a little bit.â€ He noted that many staffers of the American Embassy in Paris attended the march, including the ambassador. He said he had wanted to be present at the march himself but could not because of his prior commitments in India.â€œBut that is why I am going there on the way home, to make it crystal clear how passionately we feel about the events that have taken place there,â€ he said.â€œAnd I donâ€™t think the people of France have any doubts about Americaâ€™s understanding of what happened, of our personal sense of loss and our deep commitment to the people of France in this moment of trauma.â€")
fake_news_data("Share This Baylee Luciani (left), Screenshot of what Baylee caught on FaceTime (right) The closest Baylee Luciani could get to her boyfriend, whoâ€™s attending college in Austin, was through video online chat. The couple had regular â€œdatesâ€ this way to bridge the 200-mile distance between them. However, the endearing arrangement quickly came to an end after his FaceTime was left on and caught something that left his girlfriend horrified. Baylee had been discussing regular things with her boyfriend, Yale Gerstein, who was on the other side of the screen on an otherwise average evening. This video chat was not unlike all the others she had with Yale from his apartment near Austin Community College until the 19-year-old girlfriend heard some scratching sounds after FaceTime had been left on. According to KRON , Baylee was mid-conversation with Yale when scratches at the door caught both of their attention and he got up from his bed, where the computer was, to see who was at his door. He barely turned the handle to open in when masked men entered the room and beat Yaleâ€™s face in and slammed him down on his bed while shoving a pistol in his cheek. The intruders didnâ€™t seem to know or care that FaceTime was still on and Bayleeâ€™s face, seen in the corner, was watching everything, terrified that she was about to see her boyfriend murdered in front of her, as she watched him fight for his life. Admitting that she first thought it was a joke, seconds later, she came to the horrid realization that he was being robbed and called her dad, who was at home with her in Dallas, into the room. â€œI was scared, because they were saying Iâ€™m going to blow your head off, Iâ€™m going to kill you,â€ Baylee explained along with the chilling feeling she got when the intruder finally realized the video chat was running and looked right at her in the camera. â€œIâ€™m like wowâ€¦ seriously watching an armed robbery happen to somebody that I care about,â€ she added. Screengrabs of intruder forcing Yale down on his bed while Baylee and her father watch on FaceTime in horror With a clear view of at least one intruderâ€™s face, Baylee began taking screenshots of the suspect in the act as she and her dad called the police to report what was going on. She got the pictures right in time since, seconds later, the intruder decided to disconnect the computer as he and the suspects took off with thousands of dollars worth of Yaleâ€™s music equipment. Although the boyfriendâ€™s life was spared in the traumatizing ordeal for the two of them, he said that the thieves took something from him that canâ€™t be replaced.â€œI had just finished my first album as a solo artist,â€ Yale said. â€œThatâ€™s all lost,â€ since they took the recordings on the equipment, which means nothing to the thieves and everything to the victim. Itâ€™s not often that you hear of FaceTime solving crimes or potentially saving lives, which is what happened in this case. Although it was difficult to watch, Baylee, being there through technology, was an instrumental part in protecting Yale, who hopefully learned that he better take advantage of Texasâ€™ great gun laws and arm himself with more than just a computer.")
