
# Steganography

In this folder resides 4 steganographic methods for enbedding cover images with random messages up to a a threshold value representing bits per pixel.

They are written in separate classes, each containing an encode mathod and a decode method. The decode methods where written for testing during development.

Batch encoding of images in a dataset is possible from main.py in the parent directory.


## LSB Methods

| Method | desc | Theoretical Base |
| ---- | --- | -- |
| Sequential LSB Replacement | LSB embedding of the simplest kind, just sequentially replacing the least significant bits and attaching a delimiter to the massage to signify the end of the messages. | |
| Randomized LSB Embedding | Uses a seeded pseudorandom number generator to create an embedding path. The message path is embedded in the first 32 pixels in the path. | Khalind and Aziz (2015) |
| LSB Matching ($\pm 1$ Embedding) | Embeds the message sequentialy by randomly add +1 or -1 to the pixel value where the LSB don't already align with the message. Encodes the message length into the first 32 pixels. | Lou et al. (2010) |
| Adaptive Steganography (Sobel Edge) | Uses Sobel gradient magnitude to rank pixels by edge strength and embed data in the strongest edge regions. Encodes the message length in the noisiest 32 pixels. | Fouroozesh & Al Ja'am (2014), Smitha and Baburaj (2018) |


## References

Khalind, O., & Aziz, B. Y. Y. (2015). LSB steganography with improved embedding efficiency and undetectability. Computer Science & Information Technology, 5(1), 89–105. https://doi.org/10.5121/csit.2015.50110
(This doi doesn't resolve properly. Use this link instead: https://researchportal.port.ac.uk/en/publications/lsb-steganography-with-improved-embedding-efficiency-and-undetect)

Fouroozesh, Z., & Al ja'am, J. (2014). Image steganography based on LSBMR using Sobel edge detection. The Third International Conference on e-Technologies and Networks for Development (ICeND2014), 141–145. https://doi.org/10.1109/icend.2014.6991368

Weiqi Luo, Fangjun Huang, & Jiwu Huang. (2010). Edge Adaptive Image Steganography Based on LSB Matching Revisited. IEEE Transactions on Information Forensics and Security, 5, 201–214. https://doi.org/10.1109/tifs.2010.2041812

Smitha, G. L., & Baburaj, E. (2018). Sobel edge detection technique implementation for image steganography analysis. Biomedical Research, 29(Special Issue). https://doi.org/10.4066/biomedicalresearch.29-17-1212
