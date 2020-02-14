"""
Creates a corpus from Wikipedia dump file.
Usage (in your bash, same directory): 
python3 make_wiki_corpus ./dump/enwiki-latest-pages-articles.xml.bz2 ./dump/wiki_en.txt
"""

import sys
from gensim.corpora import WikiCorpus
import pickle

def make_corpus(in_f, out_f):

	"""Convert Wikipedia xml dump file to text corpus"""
	with open('wiki.pkl', 'wb') as pkl_out:
		wiki = WikiCorpus(in_f)
    print("Wiki corpus retrieved.")
		
		pickle.dump(wiki, pkl_out, pickle.HIGHEST_PROTOCOL)
		print("Wiki dump saved.")

	with open('wiki.pkl', 'rb') as pkl_in:
		output = open(out_f, 'w')
		
		wiki = pickle.load(pkl_in)
		print("Picke loaded.")

		i = 0
		for text in wiki.get_texts():
			output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
			i = i + 1
			if (i % 10000 == 0):
				output.close()
				output = open(out_f, 'a')
				print('Processed ' + str(i) + ' articles')
		output.close()
		print('Processing complete!')


if __name__ == '__main__':

	if len(sys.argv) != 3:
		print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
		sys.exit(1)
	in_f = sys.argv[1]
	out_f = sys.argv[2]
	make_corpus(in_f, out_f)
