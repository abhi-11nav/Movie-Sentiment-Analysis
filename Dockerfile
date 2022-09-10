FROM continuumio/anaconda3:4.4.0
COPY . /Users/abhinav/Desktop/machinelearning/MovieSentimentAnalysis
EXPOSE 5000
WORKDIR /Users/abhinav/Desktop/machinelearning/MovieSentimentAnalysis
RUN pip install -r requirements.txt
CMD python app.py
