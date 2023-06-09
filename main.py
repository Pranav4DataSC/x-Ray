import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, set_background


# set title
# st.title('Pnumonia Classifier')
set_background('./freestock4.jpg')
html_templ = """
<div style="background-color:blue;padding:10px;">
<h1 style="color:yellow">Pnuemonia Detection AI Tool 😷</h1> 
</div>
"""
st.markdown(html_templ,unsafe_allow_html=True)
##st.write("Simple proposal to diagnose Pnumonia")
##st.sidebar.image("covid19.jpg")

# set header
st.subheader('Please upload a chest X-ray image below')

# upload file
file = st.file_uploader('',type=['jpeg','jpg','png'])

# load classifier 
model = load_model('./model/pneumonia_classifier.h5')

# load class names
with open('./model/labels.txt','r') as f:
	class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
	f.close()

# display image
if file is not None:
	image = Image.open(file).convert('RGB')
	#st.image(image, use_column_width = True)
	if st.button("Image Preview",type="primary"):
		st.image(image,width=300)

	if st.button("Diagnosis",type="primary"):
		# st.image(image,width=300)
		# classify image 
		class_name, conf_score = classify(image, model, class_names)
		# write classification 
		st.write("## {}".format(class_name))
		st.write("## Score: {}%".format(int(conf_score * 1000 )/10))
	if st.button("Disclaimer",type="primary"):
		st.markdown("Pnuemonia Detection AI Tool, like any other tool does not gurantee accuracy of results. Consult your doctor for medical advice.")
		st.markdown("Author : Pranav Joshi")
		st.markdown("All Rights Reserved(2023)")
		st.markdown("Background Image of a female doctor used under license from Freestock.com")
