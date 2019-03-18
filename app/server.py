from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

# from fastai import *
from fastai.vision import *
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import sys
import imutils
import cv2
import multipart
import base64

# path variable used by html variable
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    # read image into opencv
    image = cv2.imdecode(np.fromstring(img_bytes, np.uint8), 1)
    image = imutils.resize(image, height = 850)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # define the number of bubbles to look for and the answers to the quiz
    num_bubbles = 50
    ANSWER_KEY = {0: 2, 1: 0, 2: 4, 3: 0, 4: 1, 5: 4, 6: 3, 7: 3, 8: 3, 9: 2}

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # grab_contours is a convenience function from imutils to handle
    # the differences in outputs between various OpenCV versions
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # find the contour of our page by sorting based on area and taking the approximated contour that has 4 points
    # ensure that at least one contour was found
    if len(cnts) > 0:
    	# sort the contours according to their size in descending order
    	# cv2.contourArea finds the area of our contours
    	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    	# loop over the sorted contours
    	for c in cnts:
    		# approximate the contour
    		# cv2.arcLenth finds the contour's perimeter. 2nd argument of True
    		# specifies that that the shape is a closed contour rather than a curve
    		perimeter = cv2.arcLength(c, True)
    		# approxPolyDP approximates a contour shape to another shape with less
    		# number of vertices depending upon the precision we specify
    		approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    		# if our approximated contour has four points,
    		# then we can assume we have found the paper
    		if len(approx) == 4:
    			docCnt = approx
    			break

    # to see if model is correctly finding the four corners of the page
    # output = image.copy()
    # cv2.circle(output, (docCnt[0][0][0], docCnt[0][0][1]) , 5, (0, 0, 255), -1)
    # cv2.circle(output, (docCnt[1][0][0], docCnt[1][0][1]) , 5, (0, 0, 255), -1)
    # cv2.circle(output, (docCnt[2][0][0], docCnt[2][0][1]) , 5, (0, 0, 255), -1)
    # cv2.circle(output, (docCnt[3][0][0], docCnt[3][0][1]) , 5, (0, 0, 255), -1)
    # cv2.imshow("Circle", output)
    # cv2.waitKey(0)

    # apply a four point perspective transform to both the original image and
    # grayscale image to obtain a top-down birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # implement thresholding on transformed image
    thresh_gauss = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,13,2)
    thresh_mean = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,13,2)
    thresh_simple = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    def generate_contours(threshold):
    	"""
    	Generate a list of contours given a thresholded image.
    	"""
    	cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    	cnts = imutils.grab_contours(cnts)
    	return cnts

    cnts_gauss = generate_contours(thresh_gauss)
    cnts_mean = generate_contours(thresh_mean)
    cnts_simple = generate_contours(thresh_simple)


    def generate_question_contours(contours):
    	"""
    	Return the contours of the bubbles (ideally num_bubbles set at top of this script)
    	"""
    	# loop over the contours
    	questionContours = []
    	for contour in contours:
    		# compute the bounding box of the contour, then use the
    		# bounding box to derive the aspect ratio
    		# boundingRect fits a straight (non-rotated) rectangle that bounds the countour
    		# (x,y) is the top-left coordinate of the rectangle and (w,h) are its width and height
    		(x, y, w, h) = cv2.boundingRect(contour)
    		ar = w / float(h)
    		# taking the area of our thresholds will help us eliminate the numbers
    		# in front of the bubbles like '10' and '6'
    		area = cv2.contourArea(contour)
    		# area is crucial to get the right contours. You will probably need to adjust these parameters
    		# my photos were taken from about 1.5 feet above the piece of paper, with the paper taking up
    		# most of the photo
    		if w >= 11 and w <= 20 and h >= 11 and h <= 20 and ar >= 0.6 and ar <= 1.34 and area >= 150:
    			questionContours.append(contour)
    	return questionContours

    questionCnts_gauss = generate_question_contours(cnts_gauss)
    questionCnts_mean = generate_question_contours(cnts_mean)
    questionCnts_simple = generate_question_contours(cnts_simple)

    # Different thresholding techniques will sometimes result in a different
    # number of contours being found. I want to find 50 because I have 10 questions
    # with 5 bubbles each, so I'll pick a threshold that has bubble contour length of 50
    if len(questionCnts_gauss) == num_bubbles:
    	questionCnts = questionCnts_gauss
    elif len(questionCnts_mean) == num_bubbles:
    	questionCnts = questionCnts_mean
    elif len(questionCnts_simple) == num_bubbles:
    	questionCnts = questionCnts_simple
    else:
    	questionCnts = questionCnts_gauss

    # each question has 5 possible answers, to loop over the question in batches of 5
    # sort the question contours top-to-bottom
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0

    # I'm using num_bubbles as the stopping point so I don't get dict lookup errors
    # if I find too many contours
    for (q, i) in enumerate(np.arange(0, num_bubbles, 5)):
    	# sort the contours for the current question from
    	# left to right, then initialize the index of the bubbled answer
    	# default sort for sort_contours function is 'left-to-right'. [0] returns the contours
    	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    	bubbled = None

    	# loop over the sorted contours
    	for (j, c) in enumerate(cnts):
    		# construct a mask that reveals only the current "bubble" for the question
    		# Return a new array, mask, of given shape and type, filled with zeros
    		mask = np.zeros(thresh_gauss.shape, dtype="uint8")
    		# drawContours args: source image [mask], contours passed as a list [[c]],
    		# index of contours (to draw all contours pass -1), color, thickness (-1 is filled in I think)
    		# given mask, draw all contours on the mask of zeros, filling those pixels in with 255
    		cv2.drawContours(mask, [c], -1, 255, -1)

    		# apply the mask to the thresholded image, then
    		# count the number of non-zero pixels in the bubble area
    		mask = cv2.bitwise_and(thresh_gauss, thresh_gauss, mask=mask)
    		# countNonZero returns the number of non-zero pixels in the bubble area
    		total = cv2.countNonZero(mask)

    		# Uncomment to look at masks
    		# cv2.imshow("Mask", mask)
    		# cv2.waitKey(0)

    		# if the current total has a larger number of total
    		# non-zero pixels, then we are examining the currently bubbled-in answer
    		if bubbled is None or total > bubbled[0]:
    			bubbled = (total, j)

    	# initialize the contour color and the index of the *correct* answer
    	color = (0, 0, 255)
    	k = ANSWER_KEY[q]

    	# check to see if the bubbled answer is correct
    	# bubbled[0] is the total number of non-zero pixels, while bubbled[1] has the index we are looking for
    	if k == bubbled[1]:
    		color = (0, 255, 0)
    		correct += 1

    	# draw the outline of the correct answer on the test (paper is the four point transform of the image)
    	cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    	# Uncomment to see where all the contours are being drawn
    	# to test all the contours found for each letter ('A', 'B', etc.)
    	# cv2.drawContours(paper, [cnts[0]], -1, (255, 0, 0), 1)
    	# cv2.drawContours(paper, [cnts[1]], -1, (2, 106, 253), 1)
    	# cv2.drawContours(paper, [cnts[2]], -1, (153, 0, 102), 1)
    	# cv2.drawContours(paper, [cnts[3]], -1, (128, 0, 0), 1)
    	# cv2.drawContours(paper, [cnts[4]], -1, (2, 106, 253), 1)

    # using len(ANSWER_KEY) allows me to update the score automatically
    score = (correct / len(ANSWER_KEY)) * 100
    phrases = ['Keep trying!', 'Great Work!', 'Excellent!', 'Perfect!']
    if score <= 50:
    	phrase_ouput = phrases[0]
    elif score <= 70:
    	phrase_ouput = phrases[1]
    elif score <= 90:
    	phrase_ouput = phrases[2]
    else:
    	phrase_ouput = phrases[3]

    print("Score: {:.0f}%".format(score))
    cv2.putText(paper, "{:.0f}%".format(score), (10, 30),
    	cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(paper, phrase_ouput, (80, 30),
    	cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
    cv2.putText(paper, "{} bubbles".format(len(questionCnts)), (245, 30),
    	cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

    # Uncomment to see original and graded
    # cv2.imshow("Original", image)
    # cv2.imshow("Exam", paper)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # encode paper variable as a jpg
    retval, buffer_img = cv2.imencode('.jpg', paper)
    final = base64.b64encode(buffer_img)

    # return JSONResponse({'result': "{:.0f}%".format(score)})
    # Pass base64 encoded image 'final' to JSONResponse so client.js can eventually return the image in index.html
    return JSONResponse({'result': "{:.0f}%".format(score), 'picture': "{}".format(final)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
