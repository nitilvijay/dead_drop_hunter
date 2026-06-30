# Dead Drop Hunter
Overview:
Modern attacks such as this dont involve direct injection of malicious code into the subject's device. Even if the attacker tries to inject example through a phishing
email, the malware will flagged by the email platform or the systems anitvirus using techniques like signature matching.
How do attackers make use of the dead drops?
Attackers embed the malicious code into an images using advanced steganographic algorithms. The embeddings can be in the spatila domain or in the frequency domain.
The is image in then uploaded to a public facing S3 bucket, mostly the enterprise's own bucket which provides certain service.
This image is the dead drop. At this time, the image does no harm.
The attacker then injects a loader script into a device part of the enterprise's network. This script not flagged as it not malicious on its own.
This loader script, downloads the image from the bucket. Conveniently this is not blocked by the enterprise's firewall as it is from the enterprise's own maintained
bucket.
The loader knows how to extraxt the hidden code from the image, which then executes the malicious code.

The risk this kind of attack can be reduced if the incoming images to the bucket are scanned for embeddings.
This model's main objective is to scan for any hidden embeddings in the incoming image and flag them as clean or steg.
This process happens before the image is stored.

Model

This model is an implementation of CNN. However this does not follow the usual preprocessing and archtecture of spatial based image classification.
Standard CNNs are built to recognize objects, so they try to ignore tiny pixel noises. Steganalysis CNNs are built completely upside down to do the exact opposite: preserve and magnify weak modifications


