//  Below code is written in HLS C code

Tool required for it to run is Vitis HLS tool (for simulating the HLS code and making RTL IP)
FUrther has to use Vivado tool for Block design, RTL synthesis, implementation, and bit file generation.

Stages to implement:
STage1: 
Run the deep learning model in software platform and save the model. Load the model and extract the weights and bias parameters into numpy format and text file format. 

STage2:
Design HLS code for the DL model that you are trying. Do C synthesis in vitis HLS tool and export rtl as IP. In vitis HLS tool itself you can verify your desogn by writing C test bench. 

STage3: 
Open Vivado tool, do block design, do RTL synthesis, implementation, and bit generation. Collect three files .bit file, ,hwh file, .tcl filew and load them to ZYNQ board and create overlay there. COntrol the IP by passing test images and parameters through processor of the zynq.



