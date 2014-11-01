## CH05 Classification - Detecting Pool Answers

### Problem: Detecting Pool Answers

* Representation of Data: Examples -> text ; Labels -> 0/1 (whether is a acceptable answer)

* Algorithms: K-Nearest? Logistic Regression? Decision Tree? SVM? Naive Bayes?

* Dataset: [http://www.clearbits.net/torrents/2076-aug-2012](http://www.clearbits.net/torrents/2076-aug-2012)

### Data Preprocess: 

* Select data: with CeationData earlier than 2011 (6,000,000 posts for training)

* Format data: XML -> TAB splitted (to speed up)(so_xml_to_tsv.py)

* Select useful fields (choose_instance.py): 

	* PostType(question or answer), ParentID(which question belong to)
	
	* CreationDate(time between ask question and answer)
	
	* Score
	
	* Body(content of question / answer): 
	
	* AcceptedAnswerId

	> **Fields not been selected:**

	> * ViewCount: not highly related with answer correctness, a more seriouse problem is that we can't get view count when the new answer is submitted

	> * CommentCount: simlar with ViewCount
	
	> **Other fields:**

	> * OwnerUserId: is useful when we decide to consider feature about users
	
	> * Title: provide more information about questions, we do not use this field yet now

* **Rethinking: What is a good answer?:**

	> to decide the lable of a question-sanswer example
	
	

	 