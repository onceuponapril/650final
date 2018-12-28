# 650final Movie Recommendation System
## Author: Fanpan Zeng
1. Please install the environment from requirement.txt, and enact a virtual enviroment when you run this flask. And the data is too large for git, so please download the data folder(rating.csv,movie.csv) from:https://drive.google.com/drive/folders/1T0SwEIHTsknohBS9f37ofj32die25a0w
2. Run `python3 app.py` in the terminal when you are ready with environment.
3. On the index page, the first part is content-based filtering system, which ask user: "Pick a movie you liked, let me guess what you may like".
The second part is based on collaborative filtering, wich require user:"Rate movies you watched, let me guess what you may like (0-5)". Users can choose whatever entries they like.
4. If you havn't seen the movie you want to pick or rate, simply refresh the page, it will give you a new round of 10 movies.
