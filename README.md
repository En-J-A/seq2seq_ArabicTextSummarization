# Arabic Text Summarization 

- This project includes my work on Arabic Text Summarization Project.

- Text Summarization based on DeepLearning seq2seq models using LSTM (without attention)

- Dataset is collected by scrapping Sports new site [Kooora.com](https://www.kooora.com/).

- Data collected falls within Sports category, so it may not generalize well to other domains
- Data consists of around 35k atricles using atrilce body as text and article title as Summary label

- I used the following tools:
    - python 3.8
    - selenium : used for web scrapping
    - Flask : to build the wep application
    - tensorflow & keras : for training seq2seq model
    - Heroku : used for deployment


# Demo

You can view the application from [Here](https://msaid-arabictextsummarization.herokuapp.com/)
<br><br>Note: it may take somtime to bootup the aplication due to model size, so just wait



# ToDo:
- train for more epochs(currently 50 epoch only)
- train with more data (currently 35k samples)
- deploy another model using attention mechanisms
- deploy another model using Transformers
