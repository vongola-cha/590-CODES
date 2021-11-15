
# conda install -c conda-forge wikipedia
# conda install -c conda-forge wordcloud
# pip install wikipedia_sections

import wikipedia

# see https://meta.wikimedia.org/wiki/List_of_Wikipedias
# for langages prefixes
# wikipedia.set_lang('es') #es=spanish en=english 



#--------------------------
# USER INPUTS
#--------------------------
max_num_pages=4		#max num pages returned by wiki search
verbose=False

#------------------------
#WORD CLOUD PLOT
#------------------------
def generate_word_cloud(my_text):
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    # exit()
    # Import package
    # Define a function to plot word cloud
    def plot_cloud(wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");

    # Generate word cloud
    wordcloud = WordCloud(
        width = 3000,
        height = 2000, 
        random_state=1, 
        background_color='salmon', 
        colormap='Pastel1', 
        collocations=False,
        stopwords = None).generate(my_text)
    plot_cloud(wordcloud)
    plt.show()


#------------------------
#QUERY WIKI
#------------------------

topic='joe biden'
stop_words=['']
#--------------------------
#SEARCH FOR RELEVANT PAGES 
#--------------------------
titles=wikipedia.search(topic,results=max_num_pages)
print("TITLES=",titles)

#FUNCTION TO PRINT BASIC ABOUT WIKI PAGE
def print_info(wiki_page):
	print("-------------------------")
	print(wiki_page.title)
	print(wiki_page.url)
	# print(wiki_page.sections)

	if(verbose):
		print(wiki_page.sections)
		print(wiki_page.categories)
		print(wiki_page.html)
		print(wiki_page.images)
		print(wiki_page.content)
		print(wikipedia.summary(wiki_page.title, auto_suggest=False))
		print(wiki_page.references)
		print(wiki_page.links[0],len(page.links))

#--------------------------
#LOOP OVER TITLES
#--------------------------
num_files=0
sections=[]

for title in titles:
	try:
		page = wikipedia.page(title, auto_suggest=False)
		#print_info(page)

		sections=sections+page.sections
		num_files+=1
	except:
		print("SOMETHING WENT WRONG:", title);  

#CONVERT TO ONE LONG STRING
text=''
for string in sections:
	words=string.lower().split()
	for word in words:	
		if(word not in stop_words):
			text=text+word+' '

print("TEXT:")
print(text);  
generate_word_cloud(text)
