from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import PIL
import numpy

import os , io , sys
import numpy as np 
from PIL import Image
import cv2
import base64
from yolo_detection_images import runModel

partfishthresh = 0.5

# import run_with_ngrok from flask_ngrok to run the app using ngrok
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

#dic = {0 : 'Cat', 1 : 'Dog'}
#dic = {0 : 'daisy',1 : 'dandelion',2 : 'roses',3 : 'sunflowers',4 : 'tulips'}
dic = {0 : 'bandeng',1 : 'nila',2 : 'mujair',3 : 'laosan',4 : 'gulamah',5 : 'kembung',6 : 'kuniran'}
dic_bagian = {0 : 'kepala',1 : 'ekor',2 : 'ikan utuh'}
dic_segar = {0 : 'Sangat Segar',1 : 'Segar',2 : 'Tidak Segar'}
#model = load_model('my_model2.hdf5')
model = load_model('mlr_vgg16-jenis-ikan-49-0.9821-0.9949.h5')

model.make_predict_function()




def predict_label(img_path):
	#i = image.load_img(img_path, target_size=(180,180))
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	#i = i.reshape(1, 180,180,3)
	i = i.reshape(1, 224,224,3)
	#p = model.predict_classes(i)
	p = model.predict(i)
	p = numpy.argmax(p, axis=1)
	return dic[p[0]]

def bb_iou(boxFish, boxA):
	# y_min, y_max, x_min, x_max
	# determine the (x, y)-coordinates of the intersection rectangle
	print("AAAAA")
	print(boxA)
	xA = max(boxA[0], boxFish[0])
	yA = max(boxA[2], boxFish[2])
	xFish = min(boxA[1], boxFish[1])
	yFish = min(boxA[3], boxFish[3])
	print("xFish="+str(xFish)+" xA="+str(xA)+" yFish="+str(yFish)+" yA="+str(yA))
	if xFish < xA or yFish < yA:
	    return  0.0
	# compute the area of intersection rectangle
	interArea = max(0, xFish - xA + 1) * max(0, yFish - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxFishArea = (boxFish[1] - boxFish[0] + 1) * (boxFish[3] - boxFish[2] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	#iou = interArea / float(boxAArea + boxFishArea - interArea)
	iou = interArea / float(boxAArea)
	# return the intersection over union value
	return iou

# Algoritma cek posisi BB kepala dan ekor, harus sejajar dengan  bagian ikan utuh
def get_pusat_bb(box):
	print("get_pusat")
	print(box)
	x = round((box[2]+box[3])/2)
	y = round((box[0]+box[1])/2)
	return x,y

def jarak_rasio_kepala_ekor(boxFish, boxKepala, boxEkor):
	#yFish, xFish = get_pusat(boxFish)
	xKepala, yKepala = get_pusat_bb(boxKepala)
	xEkor, yEkor = get_pusat_bb(boxEkor)
	print("yKepala xKepala yEkor xEkor")
	print(str(yKepala)+" "+str(xKepala)+" "+str(yEkor)+" "+str(xEkor))
	# Ambil tinggi dan lebar ikan utuh untuk menentukan arah obyek ikan
	h0 = boxFish[1] - boxFish[0]
	w0 = boxFish[3] - boxFish[2]
	# Jika w>h berarti ikan landscape, rasio nya h/w
	# setiap pasangan kepala ekor dihitung rasio nya, hanya yg paling dekat dengan rasio ikan utuh yang dijadikan sebagai bagian ikan utuh
	if w0>h0 :
		rasioFish = h0/w0
	else:
		rasioFish = w0/h0
	print("rasioFish")
	print(rasioFish)
	# menghitung rasio kepala-ekor
	# dihitung dari selisih y kepala dan ekor dibagi selisih x kepala dan ekor
	h1 = abs(yKepala - yEkor)
	w1 = abs(xKepala - xEkor)
	if w0>h0 :
		rasioBagian = h1/w1
	else:
		rasioBagian = w1/h1
	print("rasioBagian")
	print(rasioBagian)
	# Hitung jarak rasio dari selisih absolut rasio ikan utuh dengan bagian di kali IOU nya
	iou_ekor = bb_iou(boxFish, boxEkor)
	iou_kepala = bb_iou(boxFish, boxKepala)

	# Hitung jarak diagonal
	rFish = (h0**2 + w0**2)**0.5
	rKepalaEkor = (h1**2 + w1**2)**0.5
	#return abs(rasioFish-rasioBagian) #* (1-iou_ekor) * (1-iou_kepala)
	jarakR = abs(rasioFish-rasioBagian)
	jarakD = ((rFish-rKepalaEkor)/rFish)
	print("jarakR="+str(jarakR))
	print("jarakD="+str(jarakD))

	# Hitung jarak antar ekor dan kepala (LEBAR) dibagi LEBAR ikan
	if w0>h0:
		jarakLebar = abs(max(boxKepala[2], boxKepala[3], boxEkor[2], boxEkor[3]) - min(boxKepala[2], boxKepala[3], boxEkor[2], boxEkor[3])) / w0
	else:
		jarakLebar = abs(max(boxKepala[0], boxKepala[1], boxEkor[0], boxEkor[1]) - min(boxKepala[0], boxKepala[1], boxEkor[0], boxEkor[1])) / h0
	print("jarakLebar="+str(jarakLebar))

	# Hitung IOU
	iouKepala = bb_iou(boxFish, boxKepala)
	iouEkor = bb_iou(boxFish, boxEkor)
	print("iouKepala*iouEkor="+str(iouKepala*iouEkor))
	#return jarakR * jarakD
	return 0.6*(2-jarakLebar) + 0.4*(2-(iouKepala*iouEkor))


##########Deteksi obyek
def deteksi_obyek(img_path, hasil_path, nama_file):
	#file = request.files['my_image'].read() ## byte file
	#print(file)
	#npimg = np.fromstring(file, np.uint8)
	#img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	#i = image.load_img(img_path, target_size=(416,416))
	#i = image.img_to_array(i)/255.0
	#i = i.reshape(1, 416,416,3)

	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img2 = cv2.imread(img_path, cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################

	img, box_hasil = runModel(img, nama_file, img2)

	cv2.imwrite(hasil_path, img)


	#img = Image.fromarray(img.astype("uint8"))
	#rawBytes = io.BytesIO()
	#img.save(rawBytes, "JPEG")
	#rawBytes.seek(0)
	#img_base64 = base64.b64encode(rawBytes.read())
	#return jsonify({'status':str(img_base64)})	
	return box_hasil


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	#return "Please subscribe  Artificial Intelligence Hub..!!!"
	return render_template("index2.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		#img_path = "static/" + img.filename	
		img_path = "static/" + img.filename	
		img.save(img_path)

			
		hasil_path = "static/hasil/det_" + img.filename +"_0.jpg"

		box_hasil = deteksi_obyek(img_path, hasil_path, img.filename)
		# Saatnya klasifikasi tiap bagian tubuh yang ditermukan
		kelas_jenis = []
		bagian = []
		print(box_hasil)
		for i in range(len(box_hasil)):
			#print(box_hasil[i][1])
			kelas_bagian = box_hasil[i][1]
			# klasifikasi kelas tiap obyek
			#objek_path = box_hasil[i][6]
			#jenis = predict_label(objek_path)
			#kelas_jenis.append(jenis)
			# Ambil BB 
			bb = box_hasil[i][3:7]
			# index kepala dan ekor yang menjadi bagian dari ikan utuh disimpan dalam array bagian: kolom 1 => ikan utuh, kolom 2 => index kepala/ekor yang menjadi bagiannya, kolom 3 = IOU
			
			# Jika kelas_bagian == 2 berarti dia kepala, lakukan pengecekan bagian tubuh
			ohe = []
			bagian_tmp = []
			if kelas_bagian == 2:
				objek_path = box_hasil[i][7]
				jenis = predict_label(objek_path)
				# one hot encoding kelas kesegaran
				ohe.append(box_hasil[i][9])
				
				for j in range(len(box_hasil)):
					# cek irisan BB kepala dan ekor terhadap ikan utuh
					if box_hasil[j][1] != 2:
						iou = bb_iou(bb, box_hasil[j][3:7])
						#print(iou)
						# kalau IOU > batas maka part tersebut bagian dari ikan utuh
						if iou >= partfishthresh:
							# [kode_ikan_utuh, kode_kepala/ekor, nomor_objek_ikan_utuh, jenis_ikan, nomor_objek_kepala/ekor, iou, nama_bagian_tubuh, BB(4 elemen)]
							idx_bagian_tubuh = box_hasil[j][1]
							bagiannya = [box_hasil[i][0], box_hasil[j][0], box_hasil[i][1], jenis, box_hasil[j][1], iou, box_hasil[j][7], box_hasil[j][8], dic_bagian[idx_bagian_tubuh], box_hasil[j][3], box_hasil[j][4], box_hasil[j][5], box_hasil[j][6]]
							bagian_tmp.append(bagiannya)
							#print("box_hasil[j][9]")
							#print(box_hasil[j])
							ohe.append(box_hasil[j][9])
							
				print("ppppp")
				print(ohe)
				# Lakukan pemeriksaan ikan utuh yang terkandung lebih dari 1 kepala atau lebih dari 1 ekor
				id_ikan_utuh = box_hasil[i][0]
				print("bagian_tmp:")
				print(bagian_tmp)
				kepala, ekor = [], []
				jum_kepala, jum_ekor = 0, 0
				for j in range(len(bagian_tmp)):
					# cek bagian tubuh yang melekat pada ikan utuh
					if id_ikan_utuh == bagian_tmp[j][0]:
						if bagian_tmp[j][4]==0:
							jum_kepala += 1
							kepala.append(bagian_tmp[j])
						if bagian_tmp[j][4]==1:
							jum_ekor += 1
							ekor.append(bagian_tmp[j])
				# jika jumlah kepala atau ekor lebih dari 1 maka lakukan pemeriksaan kepala/ekor yang benar, yg salah di keluarkan dari array bagian
				if jum_kepala>1 or jum_ekor>1:
					boxFish = bb
					jarak_terkecil = 1000
					id_kepala_benar, id_ekor_benar = -1, -1
					for k in range(len(kepala)):
						boxKepala = kepala[k][9:13]
						for m in range(len(ekor)):
							boxEkor = ekor[m][9:13]
							jarak = jarak_rasio_kepala_ekor(boxFish, boxKepala, boxEkor)
							print("jarak")
							print(jarak)
							# Untuk rasio kepala:ekor terdekat dari rasio ikan_utuh, id kepala dan ekor disimpan sebagai data yg benar di array bagian
							if jarak < jarak_terkecil:
								jarak_terkecil = jarak
								id_kepala_benar = kepala[k][1]
								id_ekor_benar = ekor[m][1]
					print("id_kepala_benar, id_ekor_benar")
					print(str(id_kepala_benar)+" "+str(id_ekor_benar))
					# lakukan penghapusan kepala/ekor yang bukan bagian dari ikan_utuh
					for k in range(len(bagian_tmp)):
						print("bagian_tmp[k][1] = "+str(bagian_tmp[k][1]))
						if bagian_tmp[k][1]==id_kepala_benar or bagian_tmp[k][1]==id_ekor_benar :
							bagian.append(bagian_tmp[k])
				else:
					for k in range(len(bagian_tmp)):
						bagian.append(bagian_tmp[k])

				kesegaran_agregasi_ikan = [0,0,0]
				print("kelas=")
				print(kesegaran_agregasi_ikan)
				# Akumulasi one hot encoding
				for j in range(len(ohe)):
					kesegaran_agregasi_ikan[0]=round(kesegaran_agregasi_ikan[0]+ohe[j][0],4)
					kesegaran_agregasi_ikan[1]=round(kesegaran_agregasi_ikan[1]+ohe[j][1],4)
					kesegaran_agregasi_ikan[2]=round(kesegaran_agregasi_ikan[2]+ohe[j][2],4)
				print("kesegaran_agregasi_ikan=")
				print(kesegaran_agregasi_ikan)
				terbesar = numpy.max(kesegaran_agregasi_ikan, axis=0)
				idx_terbesar = numpy.argmax(kesegaran_agregasi_ikan, axis=0)
				agregasi_segar = dic_segar[idx_terbesar]
				# Simpan jenis ikan pada obyek ikan utuh
				kelas_jenis.append([box_hasil[i], jenis, kesegaran_agregasi_ikan, terbesar, agregasi_segar])
				
				# Cek ikan utuh yang ada kepala saja atau ekor saja, pilih yang irisan terbesar
				if (jum_kepala>=1 and jum_ekor == 0) or (jum_kepala == 0 and jum_ekor>=1):
					idx_iou_terbesar = -1
					iou_terbesar = 0
					for j in range(len(box_hasil)):
						# cek irisan BB kepala dan ekor terhadap ikan utuh
						if box_hasil[j][1] != 2:
							iou = bb_iou(bb, box_hasil[j][3:7])
							#print(iou)
							# kalau IOU > batas maka part tersebut bagian dari ikan utuh
							if iou >= partfishthresh and iou > iou_terbesar:
								iou_terbesar = iou
								idx_iou_terbesar = j
					# Selesai iterasi, masukkan ke bagian
					idx_bagian_tubuh = box_hasil[idx_iou_terbesar][1]
					bagiannya = [box_hasil[i][0], box_hasil[idx_iou_terbesar][0], box_hasil[i][1], jenis, box_hasil[idx_iou_terbesar][1], iou, box_hasil[idx_iou_terbesar][7], box_hasil[idx_iou_terbesar][8], dic_bagian[idx_bagian_tubuh], box_hasil[idx_iou_terbesar][3], box_hasil[idx_iou_terbesar][4], box_hasil[idx_iou_terbesar][5], box_hasil[idx_iou_terbesar][6]]

					bagian.append(bagiannya)
		#p = predict_label(img_path)
		# Mengumpulkan objek yang bukan bagian dari ikan 
		bagian_lain = []
		for i in range(len(box_hasil)):
			kode_objek = box_hasil[i][0]
			# Cek hanya yg nomor_objek bukan ikan utuh
			
			if box_hasil[i][1] != 2:
				ada = False
				for j in range(len(bagian)):
					if bagian[j][1] == kode_objek:
						ada = True
				if ada == False:
					idx_bagian_tubuh = box_hasil[i][1]
					bagian_lain.append([box_hasil[i][0], box_hasil[i][1], box_hasil[i][2], box_hasil[i][3], box_hasil[i][4], box_hasil[i][5], box_hasil[i][6], box_hasil[i][7], box_hasil[i][8], dic_bagian[idx_bagian_tubuh]])
		# Lakukan perhitungan vooting kelas kesegaran pada ikan utuh dan bagian tubuh selain bagian tubuh yang manjadi ikan utuh
		vooting_kelas = [0, 0, 0]
		jum = 0
		for i in range(len(kelas_jenis)):
			jum += 1
			print("kelas_jenis[i][0][8]="+str(kelas_jenis[i][0][8]))
			if kelas_jenis[i][0][8] == "Sangat Segar":
				vooting_kelas[0] += 1
			elif kelas_jenis[i][0][8] == "Segar":
				vooting_kelas[1] += 1
			else:
				vooting_kelas[2] += 1
		for j in range(len(bagian_lain)):
			jum += 1
			print("bagian_lain[j][8]="+str(bagian_lain[j][8]))
			if bagian_lain[j][8] == "Sangat Segar":
				vooting_kelas[0] += 1
			elif bagian_lain[j][8] == "Segar":
				vooting_kelas[1] += 1
			else:
				vooting_kelas[2] += 1
		# Berikan kesimpulan
		print("vooting kelas = "+str(vooting_kelas))
		nilai_mayor_1 = np.max(vooting_kelas)
		idx_kelas_mayor_1 = np.argmax(vooting_kelas)
		kelas_mayor_1 = dic_segar[idx_kelas_mayor_1]
		vooting_kelas[idx_kelas_mayor_1] = -1
		nilai_mayor_2 = np.max(vooting_kelas)
		idx_kelas_mayor_2 = np.argmax(vooting_kelas)
		kelas_mayor_2 = dic_segar[idx_kelas_mayor_2]
		if nilai_mayor_1/jum >= 0.8:
			simpulan = "Mayoritas atau semua ikan dan bagian tubuhnya adalah " + kelas_mayor_1
		elif nilai_mayor_1/jum <= 0.5 or nilai_mayor_1/jum < 0.8:
			simpulan = "Sebagian besar ikan utuh dan bagian tubuh adalah " + kelas_mayor_1 + ", sebagian kecil lainnya " + kelas_mayor_2
		else:
			simpulan = "Tidak ditemukan kesimpulan"
	print("ikan utuh yang ditemukan")
	print(kelas_jenis)
	print("bagian ikan utuh yang ditemukan")
	print(bagian)
	print("bagian tubuh lainnya")
	print(bagian_lain)
	print(len(kelas_jenis))
	
	return render_template("index.html", prediction = kelas_jenis, len = len(kelas_jenis), img_path = img_path, hasil_path = hasil_path, bagian_lain = bagian_lain, len2 = len(bagian_lain), bagian = bagian, len3 = len(bagian), simpulan = simpulan)


if __name__ =='__main__':
	#app.debug = True
	#app.run(debug = True)
	app.run()