#include <stdio.h>
#include "layer_lib.h"

#define INPUT_SIZE 1
#define HIDDEN_SIZE1 64
#define HIDDEN_SIZE2 32
#define OUTPUT_SIZE 1
#include <math.h>
#include "datatype.h"

inline d16 sigmoid_d16(d16 x) {	
    if (x <= -2.5) {
        return 0;
    } else if (x >= 2.5) {
        return 1;
    } else {
        return 0.2 * x + 0.5;
    }
}
//inline d16 hard_tanh(d16 x) {
inline d16 tanh_d16(d16 x) {
    if (x <= -1) {
        return -1;
    } else if (x >= 1) {
        return 1;
    } else {
        return x;
    }
}

void lstm_acc(d16 *ly1_inpw, d16 *ly1_rw, d16 *ly1_b, d16 *ly2_inpw, d16 *ly2_rw, d16 *ly2_b, d16 *dw, d16 *inputx, d16 *outputx);
void cnn_acc(d16 *conv1_w, d16 *conv1_bias,  d16 *conv2_w, d16 *conv2_bias, d16 *conv3_w, d16 *conv3_bias, d16 *fc1u_w, d16 *fc1v_w, d16 *fc1_bias, d16 *fc2u_w, d16 *fc2v_w, d16 *fc2_bias, d16 *fc3_w, d16 *fc3_bias, d16 *input_data, d16 *output_data);

void  hybrid_acc3(
    d16 *conv1_w,
    d16 *conv1_bias,
    d16 *conv2_w,
    d16 *conv2_bias,
    d16 *conv3_w,
    d16 *conv3_bias,    
    d16 *fc1u_w,
    d16 *fc1v_w,
    d16 *fc1_bias,
    d16 *fc2u_w,
    d16 *fc2v_w,
    d16 *fc2_bias,
    d16 *fc3_w,
    d16 *fc3_bias,
    d16 *cnn_input,
    d16 *cnn_output,
    
			d16 *ly1_inpw,
			d16 *ly1_rw,
			d16 *ly1_b,
			d16 *ly2_inpw,
			d16 *ly2_rw,
			d16 *ly2_b,	
			d16 *dw,    
			d16 *lstm_input,
			d16 *lstm_output			
)
{

#pragma HLS INTERFACE m_axi depth=256 port=cnn_output offset=slave bundle=cnn_out
#pragma HLS INTERFACE s_axilite port=cnn_output
#pragma HLS INTERFACE m_axi depth=256 port=cnn_input offset=slave bundle=cnn_inp
#pragma HLS INTERFACE s_axilite port=cnn_input
#pragma HLS INTERFACE m_axi depth=256 port=fc3_w offset=slave bundle=fc3w
#pragma HLS INTERFACE s_axilite port=fc3_w
#pragma HLS INTERFACE m_axi depth=256 port=fc3_bias offset=slave bundle=fc3b
#pragma HLS INTERFACE s_axilite port=fc3_bias
#pragma HLS INTERFACE m_axi depth=256 port=fc2u_w offset=slave bundle=fc2uw
#pragma HLS INTERFACE s_axilite port=fc2u_w
#pragma HLS INTERFACE m_axi depth=256 port=fc2v_w offset=slave bundle=fc2vw
#pragma HLS INTERFACE s_axilite port=fc2v_w
#pragma HLS INTERFACE m_axi depth=256 port=fc2_bias offset=slave bundle=fc2b
#pragma HLS INTERFACE s_axilite port=fc2_bias
#pragma HLS INTERFACE m_axi depth=256 port=fc1u_w offset=slave bundle=fc1uw
#pragma HLS INTERFACE s_axilite port=fc1u_w
#pragma HLS INTERFACE m_axi depth=256 port=fc1v_w offset=slave bundle=fc1vw
#pragma HLS INTERFACE s_axilite port=fc1v_w
#pragma HLS INTERFACE m_axi depth=256 port=fc1_bias offset=slave bundle=fc1b
#pragma HLS INTERFACE s_axilite port=fc1_bias
#pragma HLS INTERFACE m_axi depth=256 port=conv2_w offset=slave bundle=con2w
#pragma HLS INTERFACE s_axilite port=conv2_w
#pragma HLS INTERFACE m_axi depth=256 port=conv2_bias offset=slave bundle=con2b
#pragma HLS INTERFACE s_axilite port=conv2_bias
#pragma HLS INTERFACE m_axi depth=256 port=conv3_w offset=slave bundle=con3w
#pragma HLS INTERFACE s_axilite port=conv3_w
#pragma HLS INTERFACE m_axi depth=256 port=conv3_bias offset=slave bundle=con3b
#pragma HLS INTERFACE s_axilite port=conv3_bias
#pragma HLS INTERFACE m_axi depth=256 port=conv1_w offset=slave bundle=con1w
#pragma HLS INTERFACE s_axilite port=conv1_w
#pragma HLS INTERFACE m_axi depth=256 port=conv1_bias offset=slave bundle=con1b
#pragma HLS INTERFACE s_axilite port=conv1_bias


#pragma HLS INTERFACE m_axi depth=256 port=lstm_input offset=slave bundle=lstm_inp
#pragma HLS INTERFACE s_axilite port=lstm_input
#pragma HLS INTERFACE m_axi depth=256 port=lstm_output offset=slave bundle=lstm_out
#pragma HLS INTERFACE s_axilite port=lstm_output
#pragma HLS INTERFACE m_axi depth=256 port=ly1_inpw offset=slave bundle=ly1w
#pragma HLS INTERFACE s_axilite port=ly1_inpw
#pragma HLS INTERFACE m_axi depth=256 port=ly1_rw offset=slave bundle=ly1rw
#pragma HLS INTERFACE s_axilite port=ly1_rw
#pragma HLS INTERFACE m_axi depth=256 port=ly1_b offset=slave bundle=ly1b
#pragma HLS INTERFACE s_axilite port=ly1_b
#pragma HLS INTERFACE m_axi depth=256 port=ly2_inpw offset=slave bundle=ly2w
#pragma HLS INTERFACE s_axilite port=ly2_inpw
#pragma HLS INTERFACE m_axi depth=256 port=ly2_rw offset=slave bundle=ly2rw
#pragma HLS INTERFACE s_axilite port=ly2_rw
#pragma HLS INTERFACE m_axi depth=256 port=ly2_b offset=slave bundle=ly2b
#pragma HLS INTERFACE s_axilite port=ly2_b
#pragma HLS INTERFACE m_axi depth=256 port=dw offset=slave bundle=denw
#pragma HLS INTERFACE s_axilite port=dw

#pragma HLS INTERFACE s_axilite port=return

cnn_acc(conv1_w, conv1_bias, conv2_w, conv2_bias, conv3_w, conv3_bias, fc1u_w, fc1v_w, fc1_bias, fc2u_w, fc2v_w, fc2_bias, fc3_w, fc3_bias, cnn_input, cnn_output);
lstm_acc(ly1_inpw, ly1_rw, ly1_b, ly2_inpw, ly2_rw, ly2_b, dw, lstm_input, lstm_output);

return;
}
    

void lstm_acc(d16 *ly1_inpw, d16 *ly1_rw, d16 *ly1_b, d16 *ly2_inpw, d16 *ly2_rw, d16 *ly2_b, d16 *dw, d16 *inputx, d16 *outputx)
{         

d16 input_seq[3];
d16 input;
d16 lstm1_input_weights[INPUT_SIZE][4 * HIDDEN_SIZE1];
d16 lstm1_recurrent_weights[HIDDEN_SIZE1][4 * HIDDEN_SIZE1];
#pragma HLS ARRAY_PARTITION variable=lstm1_recurrent_weights cyclic factor=8 dim=1
d16 lstm1_bias[4 * HIDDEN_SIZE1];
d16 lstm2_input_weights[HIDDEN_SIZE1][4 * HIDDEN_SIZE2];
#pragma HLS ARRAY_PARTITION variable=lstm2_input_weights cyclic factor=8 dim=1
d16 lstm2_recurrent_weights[HIDDEN_SIZE2][4 * HIDDEN_SIZE2];
#pragma HLS ARRAY_PARTITION variable=lstm2_recurrent_weights cyclic factor=4 dim=1
d16 lstm2_bias[4 * HIDDEN_SIZE2];
d16 dense_weights[HIDDEN_SIZE2][OUTPUT_SIZE];
d16 dense_bias[OUTPUT_SIZE] = {0.04070737};

Loop_x1:for(int x=0;x<256;x++)
{
		lstm1_input_weights[0][x] =  *ly1_inpw++;				
}
Loop_x2:for(int x=0;x<64;x++)
{
	Loop_x3:for(int y=0;y<256;y++)
	{
		lstm1_recurrent_weights[x][y] =  *ly1_rw++;	
	}
	Loop_x4:for(int y=0;y<128;y++)
	{
		lstm2_input_weights[x][y] =  *ly2_inpw++;	
	}	
}
Loop_x5:for(int x=0;x<32;x++)
{
	Loop_x6:for(int y=0;y<128;y++)
	{
		lstm2_recurrent_weights[x][y] =  *ly2_rw++;	
	}
}


Loop_x7:for(int x=0;x<256;x++)
{
	lstm1_bias[x] = *ly1_b++;
}
Loop_x8:for(int x=0;x<128;x++)
{
	lstm2_bias[x] = *ly2_b++;	
}

Loop_x9:for(int x=0;x<32;x++)
{
dense_weights[x][0] = *dw++;
}
    static d16 c1[HIDDEN_SIZE1] = {0.0};
    static d16 h1[HIDDEN_SIZE1] = {0.0};
    static d16 c2[HIDDEN_SIZE2] = {0.0};
    static d16 h2[HIDDEN_SIZE2] = {0.0};
    #pragma HLS ARRAY_PARTITION variable=c1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=h1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=c2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=h2 complete dim=0
    
    
	d16 input_gate_tmpx[HIDDEN_SIZE1];
	d16 forget_gate_tmpx[HIDDEN_SIZE1];
	d16 cell_state_tmpx[HIDDEN_SIZE1];
	d16 output_gate_tmpx[HIDDEN_SIZE1];
	#pragma HLS ARRAY_PARTITION variable=input_gate_tmpx complete dim=0
	#pragma HLS ARRAY_PARTITION variable=forget_gate_tmpx complete dim=0
	#pragma HLS ARRAY_PARTITION variable=cell_state_tmpx complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_gate_tmpx complete dim=0
	d16 input_gate_tmpy[HIDDEN_SIZE2];
	d16 forget_gate_tmpy[HIDDEN_SIZE2];
	d16 cell_state_tmpy[HIDDEN_SIZE2];
	d16 output_gate_tmpy[HIDDEN_SIZE2];
	#pragma HLS ARRAY_PARTITION variable=input_gate_tmpy complete dim=0
	#pragma HLS ARRAY_PARTITION variable=forget_gate_tmpy complete dim=0
	#pragma HLS ARRAY_PARTITION variable=cell_state_tmpy complete dim=0
	#pragma HLS ARRAY_PARTITION variable=output_gate_tmpy complete dim=0	
	
	d16 a1input_gate_tmp;
	d16 a1forget_gate_tmp;
	d16 a1cell_state_tmp;
	d16 a1output_gate_tmp;  
	
	d16 b1input_gate_tmp;
	d16 b1forget_gate_tmp;
	d16 b1cell_state_tmp;
	d16 b1output_gate_tmp;  
	
	d16 c1input_gate_tmp;
	d16 c1forget_gate_tmp;
	d16 c1cell_state_tmp;
	d16 c1output_gate_tmp;  
	
	d16 d1input_gate_tmp;
	d16 d1forget_gate_tmp;
	d16 d1cell_state_tmp;
	d16 d1output_gate_tmp;  		
	
	d16 e1input_gate_tmp;
	d16 e1forget_gate_tmp;
	d16 e1cell_state_tmp;
	d16 e1output_gate_tmp;  
	
	d16 f1input_gate_tmp;
	d16 f1forget_gate_tmp;
	d16 f1cell_state_tmp;
	d16 f1output_gate_tmp;  
	
	d16 g1input_gate_tmp;
	d16 g1forget_gate_tmp;
	d16 g1cell_state_tmp;
	d16 g1output_gate_tmp;  
	
	d16 h1input_gate_tmp;
	d16 h1forget_gate_tmp;
	d16 h1cell_state_tmp;
	d16 h1output_gate_tmp;  					
	
	d16 i1input_gate_tmp;
	d16 i1forget_gate_tmp;
	d16 i1cell_state_tmp;
	d16 i1output_gate_tmp;  					
	
	d16 j1input_gate_tmp;
	d16 j1forget_gate_tmp;
	d16 j1cell_state_tmp;
	d16 j1output_gate_tmp;  		
	
	d16 k1input_gate_tmp;
	d16 k1forget_gate_tmp;
	d16 k1cell_state_tmp;
	d16 k1output_gate_tmp;  				
			
	d16 ly2xinput_gate_tmp;
	d16 ly2xforget_gate_tmp;
	d16 ly2xcell_state_tmp;
	d16 ly2xoutput_gate_tmp;   
	d16 ly2yinput_gate_tmp;
	d16 ly2yforget_gate_tmp;
	d16 ly2ycell_state_tmp;
	d16 ly2youtput_gate_tmp;   		 	
	
    	d16 input_gate[HIDDEN_SIZE1];
    	d16 forget_gate[HIDDEN_SIZE1];
    	d16 cell_state[HIDDEN_SIZE1];
    	d16 output_gate[HIDDEN_SIZE1];
    	
    	d16 input_gate2[HIDDEN_SIZE2];
    	d16 forget_gate2[HIDDEN_SIZE2];
    	d16 cell_state2[HIDDEN_SIZE2];
    	d16 output_gate2[HIDDEN_SIZE2];    	
		
    
    Loop_x10:for (int k = 0; k < 55; k++) 
	{
        input = *inputx++;


    	Loop_x11:for (int i = 0; i < HIDDEN_SIZE1; i++) 
		{
			#pragma HLS pipeline II=16
        	a1input_gate_tmp = 0.0;
        	a1forget_gate_tmp = 0.0;	
        	a1cell_state_tmp = 0.0;
        	a1output_gate_tmp = 0.0;
        	
        	b1input_gate_tmp = 0.0;
        	b1forget_gate_tmp = 0.0;	
        	b1cell_state_tmp = 0.0;
        	b1output_gate_tmp = 0.0;
			
        	c1input_gate_tmp = 0.0;
        	c1forget_gate_tmp = 0.0;	
        	c1cell_state_tmp = 0.0;
        	c1output_gate_tmp = 0.0;
			
        	d1input_gate_tmp = 0.0;
        	d1forget_gate_tmp = 0.0;	
        	d1cell_state_tmp = 0.0;
        	d1output_gate_tmp = 0.0;	
			
        	e1input_gate_tmp = 0.0;
        	e1forget_gate_tmp = 0.0;	
        	e1cell_state_tmp = 0.0;
        	e1output_gate_tmp = 0.0;
			
        	f1input_gate_tmp = 0.0;
        	f1forget_gate_tmp = 0.0;	
        	f1cell_state_tmp = 0.0;
        	f1output_gate_tmp = 0.0;
			
        	g1input_gate_tmp = 0.0;
        	g1forget_gate_tmp = 0.0;	
        	g1cell_state_tmp = 0.0;
        	g1output_gate_tmp = 0.0;
			
        	h1input_gate_tmp = 0.0;
        	h1forget_gate_tmp = 0.0;	
        	h1cell_state_tmp = 0.0;
        	h1output_gate_tmp = 0.0;	
			
        	i1input_gate_tmp = 0.0;
        	i1forget_gate_tmp = 0.0;	
        	i1cell_state_tmp = 0.0;
        	i1output_gate_tmp = 0.0;
			
        	j1input_gate_tmp = 0.0;
        	j1forget_gate_tmp = 0.0;	
        	j1cell_state_tmp = 0.0;
        	j1output_gate_tmp = 0.0;
			
        	k1input_gate_tmp = 0.0;
        	k1forget_gate_tmp = 0.0;	
        	k1cell_state_tmp = 0.0;
        	k1output_gate_tmp = 0.0;																									        	
        	
        	Loop_lyx1: for(int j=0;j<HIDDEN_SIZE1;j++)
        	{
        		#pragma HLS unroll factor=8
				input_gate_tmpx[j] = 0.0;
				forget_gate_tmpx[j] = 0.0;
				cell_state_tmpx[j] = 0.0;
				output_gate_tmpx[j] = 0.0;        		
			}

        	Loop_x12:for (int j = 0; j < INPUT_SIZE; j++) {
            	a1input_gate_tmp += lstm1_input_weights[j][i] * input + lstm1_bias[i];;
            	a1forget_gate_tmp += lstm1_input_weights[j][i + 64] * input + lstm1_bias[i + 64];;
            	a1cell_state_tmp += lstm1_input_weights[j][i + 128] * input + lstm1_bias[i + 128];;
            	a1output_gate_tmp += lstm1_input_weights[j][i + 192] * input + lstm1_bias[i + 192];;
        	}

        	Loop_x13:for (int j = 0; j < HIDDEN_SIZE1; j++) {
        		#pragma HLS unroll factor=8
            	input_gate_tmpx[j] = lstm1_recurrent_weights[j][i] * h1[j];
            	forget_gate_tmpx[j] = lstm1_recurrent_weights[j][i + 64] * h1[j];
            	cell_state_tmpx[j] = lstm1_recurrent_weights[j][i + 128] * h1[j];
            	output_gate_tmpx[j] = lstm1_recurrent_weights[j][i + 192] * h1[j];
        	}
        	
        	Loop_lyy1: for(int j=0;j<HIDDEN_SIZE1;j = j+8)
        	{
				a1input_gate_tmp +=input_gate_tmpx[j];
				a1forget_gate_tmp +=forget_gate_tmpx[j];
				a1cell_state_tmp +=cell_state_tmpx[j];
				a1output_gate_tmp +=output_gate_tmpx[j];    
				
				b1input_gate_tmp +=input_gate_tmpx[j+1];
				b1forget_gate_tmp +=forget_gate_tmpx[j+1];
				b1cell_state_tmp +=cell_state_tmpx[j+1];
				b1output_gate_tmp +=output_gate_tmpx[j+1];   
				
				c1input_gate_tmp +=input_gate_tmpx[j+2];
				c1forget_gate_tmp +=forget_gate_tmpx[j+2];
				c1cell_state_tmp +=cell_state_tmpx[j+2];
				c1output_gate_tmp +=output_gate_tmpx[j+2];  
				
				d1input_gate_tmp +=input_gate_tmpx[j+3];
				d1forget_gate_tmp +=forget_gate_tmpx[j+3];
				d1cell_state_tmp +=cell_state_tmpx[j+3];
				d1output_gate_tmp +=output_gate_tmpx[j+3];   
				
				e1input_gate_tmp +=input_gate_tmpx[j+4];
				e1forget_gate_tmp +=forget_gate_tmpx[j+4];
				e1cell_state_tmp +=cell_state_tmpx[j+4];
				e1output_gate_tmp +=output_gate_tmpx[j+4]; 
				
				f1input_gate_tmp +=input_gate_tmpx[j+5];
				f1forget_gate_tmp +=forget_gate_tmpx[j+5];
				f1cell_state_tmp +=cell_state_tmpx[j+5];
				f1output_gate_tmp +=output_gate_tmpx[j+5]; 
				
				g1input_gate_tmp +=input_gate_tmpx[j+6];
				g1forget_gate_tmp +=forget_gate_tmpx[j+6];
				g1cell_state_tmp +=cell_state_tmpx[j+6];
				g1output_gate_tmp +=output_gate_tmpx[j+6]; 
				
				h1input_gate_tmp +=input_gate_tmpx[j+7];
				h1forget_gate_tmp +=forget_gate_tmpx[j+7];
				h1cell_state_tmp +=cell_state_tmpx[j+7];
				h1output_gate_tmp +=output_gate_tmpx[j+7]; 																				 								    		
			}        	
			i1input_gate_tmp	=	a1input_gate_tmp	+ 	b1input_gate_tmp	+	c1input_gate_tmp	+	d1input_gate_tmp;	 
			j1input_gate_tmp	=	e1input_gate_tmp	+	f1input_gate_tmp	+	g1input_gate_tmp	+	h1input_gate_tmp;
			i1forget_gate_tmp	=	a1forget_gate_tmp	+	b1forget_gate_tmp	+	c1forget_gate_tmp	+	d1forget_gate_tmp;
			j1forget_gate_tmp	=	e1forget_gate_tmp	+	f1forget_gate_tmp	+	g1forget_gate_tmp	+	h1forget_gate_tmp;
			i1cell_state_tmp	=	a1cell_state_tmp	+	b1cell_state_tmp	+	c1cell_state_tmp	+	d1cell_state_tmp;
			j1cell_state_tmp	=	e1cell_state_tmp	+	f1cell_state_tmp	+	g1cell_state_tmp	+	h1cell_state_tmp;
			i1output_gate_tmp	=	a1output_gate_tmp	+	b1output_gate_tmp	+	c1output_gate_tmp	+	d1output_gate_tmp;
			j1output_gate_tmp	=	e1output_gate_tmp	+	f1output_gate_tmp	+	g1output_gate_tmp	+	h1output_gate_tmp;
			
			k1input_gate_tmp	=	i1input_gate_tmp	+ 	j1input_gate_tmp;
			k1forget_gate_tmp	=	i1forget_gate_tmp	+	j1forget_gate_tmp;
			k1cell_state_tmp	=	i1cell_state_tmp	+	j1cell_state_tmp;
			k1output_gate_tmp	=	i1output_gate_tmp	+	j1output_gate_tmp;

        		input_gate[i] = sigmoid_d16(	k1input_gate_tmp);
        		forget_gate[i] = sigmoid_d16(	k1forget_gate_tmp);
        		cell_state[i] = tanh_d16(		k1cell_state_tmp);
        		output_gate[i] = sigmoid_d16(	k1output_gate_tmp);
    	}
    	Loop_x14:for(int i=0;i<64;i++)
    	{
    			c1[i] = forget_gate[i] * c1[i] + input_gate[i] * cell_state[i];
        		h1[i] = output_gate[i] * tanh_d16(c1[i]);
		}



    	Loop_x15:for (int i = 0; i < HIDDEN_SIZE2; i++) 
		{
			#pragma HLS pipeline II=16
        	ly2xinput_gate_tmp = 0.0;
        	ly2xforget_gate_tmp = 0.0;
        	ly2xcell_state_tmp = 0.0;
        	ly2xoutput_gate_tmp = 0.0;
        	
        	ly2yinput_gate_tmp = 0.0;
        	ly2yforget_gate_tmp = 0.0;
        	ly2ycell_state_tmp = 0.0;
        	ly2youtput_gate_tmp = 0.0;        	
        	
        	Loop_lyx21: for(int j=0;j<HIDDEN_SIZE1;j++)
        	{
        		#pragma HLS unroll
				input_gate_tmpx[j] = 0.0;
				forget_gate_tmpx[j] = 0.0;
				cell_state_tmpx[j] = 0.0;
				output_gate_tmpx[j] = 0.0;        		
			}    
        	Loop_lyx22: for(int j=0;j<HIDDEN_SIZE2;j++)
        	{
        		#pragma HLS unroll
				input_gate_tmpy[j] = 0.0;
				forget_gate_tmpy[j] = 0.0;
				cell_state_tmpy[j] = 0.0;
				output_gate_tmpy[j] = 0.0;        		
			}  			    	

        	Loop_x16:for (int j = 0; j < HIDDEN_SIZE1; j++) {
        		#pragma HLS unroll factor=8
            	input_gate_tmpx[j] = lstm2_input_weights[j][i] * h1[j];
            	forget_gate_tmpx[j] = lstm2_input_weights[j][i + 32] * h1[j];
            	cell_state_tmpx[j] = lstm2_input_weights[j][i + 64] * h1[j];
            	output_gate_tmpx[j] = lstm2_input_weights[j][i + 128] * h1[j];
        	}

        	Loop_x17:for (int j = 0; j < HIDDEN_SIZE2; j++) {
        		#pragma HLS unroll factor=4
            	input_gate_tmpy[j] = lstm2_recurrent_weights[j][i] * h2[j];
            	forget_gate_tmpy[j] = lstm2_recurrent_weights[j][i + 32] * h2[j];
            	cell_state_tmpy[j] = lstm2_recurrent_weights[j][i + 64] * h2[j];
            	output_gate_tmpy[j] = lstm2_recurrent_weights[j][i + 128] * h2[j];
        	}

        	ly2xinput_gate_tmp += lstm2_bias[i];
        	ly2xforget_gate_tmp += lstm2_bias[i + 32];
        	ly2xcell_state_tmp += lstm2_bias[i + 64];
        	ly2xoutput_gate_tmp += lstm2_bias[i + 128];
        	
        	Loop_lyy21: for(int j=0;j<HIDDEN_SIZE1;j++)
        	{
					ly2xinput_gate_tmp +=input_gate_tmpx[j];
					ly2xforget_gate_tmp +=forget_gate_tmpx[j];
					ly2xcell_state_tmp +=cell_state_tmpx[j];
					ly2xoutput_gate_tmp +=output_gate_tmpx[j];
			}
        	Loop_lyy22: for(int j=0;j<HIDDEN_SIZE2;j++)
        	{        		
					ly2yinput_gate_tmp +=input_gate_tmpy[j];
					ly2yforget_gate_tmp +=forget_gate_tmpy[j];
					ly2ycell_state_tmp +=cell_state_tmpy[j];
					ly2youtput_gate_tmp +=output_gate_tmpy[j];         		
        	}
        	
        	input_gate2[i] = sigmoid_d16(ly2xinput_gate_tmp + 	ly2yinput_gate_tmp);
        	forget_gate2[i] = sigmoid_d16(ly2xforget_gate_tmp + 	ly2yforget_gate_tmp);	
        	cell_state2[i] = tanh_d16(ly2xcell_state_tmp + 		ly2ycell_state_tmp);
        	output_gate2[i] = sigmoid_d16(ly2xoutput_gate_tmp + 	ly2youtput_gate_tmp);
    	}
    
    	Loop_x18:for(int i=0;i<32;i++)
    	{
    		c2[i] = forget_gate2[i] * c2[i] + input_gate2[i] * cell_state2[i];
        	h2[i] = output_gate2[i] * tanh_d16(c2[i]);
		}	
	}

    d16 dense_out = 0.0;
    Loop_x19:for (int i = 0; i < HIDDEN_SIZE2; i++) {
        dense_out += h2[i] * dense_weights[i][0];
    }
    dense_out += dense_bias[0];
    *outputx = dense_out;
				     
     
}

void cnn_acc(d16 *conv1_w, d16 *conv1_bias,  d16 *conv2_w, d16 *conv2_bias, d16 *conv3_w, d16 *conv3_bias, d16 *fc1u_w, d16 *fc1v_w, d16 *fc1_bias, d16 *fc2u_w, d16 *fc2v_w, d16 *fc2_bias, d16 *fc3_w, d16 *fc3_bias, d16 *input_data, d16 *output_data)
{
   
    /****************************************
     ********* REGISTER DEFINITION**************
     *********************************/
    d16 input_data_buf[input_channel][input_R][input_C];
//#pragma HLS ARRAY_PARTITION variable=input_data_buf complete dim=1
    d16 conv1_out[CONV1_CHout][CONV1_R][CONV1_C];
#pragma HLS ARRAY_PARTITION variable=conv1_out cyclic factor=8 dim=1
    d16 pooling1_out[POOLING1_CHout][POOLING1_R][POOLING1_C+2];
//#pragma HLS ARRAY_PARTITION variable=pooling1_out cyclic factor=8 dim=1
    d16 conv2_out[CONV2_CHout][CONV2_R][CONV2_C];
#pragma HLS ARRAY_PARTITION variable=conv2_out cyclic factor=8 dim=1
    d16 pooling2_out[POOLING2_CHout][POOLING2_R][POOLING2_C+2];
    d16 conv3_out[CONV3_CHout][CONV3_R][CONV3_C];
    #pragma HLS ARRAY_PARTITION variable=conv3_out cyclic factor=8 dim=1
    d16 pooling3_out_buf[POOLING2_CHout][POOLING2_R][POOLING2_C];
    d16 pooling3_out[POOLING3_CHout*POOLING3_C*POOLING3_R];
    d16 fc1_out[FC1_OUTPUT_NUM1];
    d16 fc2_out[FC2_OUTPUT_NUM1];
    d16 output_data_buf[output_length];
    d16 fc1_weights_buf[FC1_INPUT_NUM1];
    d16 fc1_bias_buf[FC1_OUTPUT_NUM1];
    d16 conv1_W_buf[CONV1_CHout][CONV1_CHin][1][CONV_K];
//#pragma HLS ARRAY_PARTITION variable=conv1_W_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv1_W_buf cyclic factor=8 dim=1
    d16 conv1_bias_buf[CONV1_CHout];
    d16 conv2_W_buf[CONV2_CHout][CONV2_CHin];
#pragma HLS ARRAY_PARTITION variable=conv2_W_buf cyclic factor=8 dim=1
d16 conv3_W_buf[CONV3_CHout][CONV3_CHin];
#pragma HLS ARRAY_PARTITION variable=conv3_W_buf cyclic factor=8 dim=1
    d16 conv2_bias_buf[CONV2_CHout];
    d16 conv3_bias_buf[CONV3_CHout];
    d16 fc2_weights_buf[FC2_INPUT_NUM1];
    d16 fc2_bias_buf[FC2_OUTPUT_NUM1];
    d16 fc3_weights_buf[FC3_INPUT_NUM1];
    d16 fc3_bias_buf[FC3_OUTPUT_NUM1];
    
    d16 fc1_outu[6];
    d16 fc2_outu[6];

    /****************************************
     ********* LOAD SOME WEIGHTS**************
     *********************************/
    Loop_rd1:for (int i = 0; i < FC1_OUTPUT_NUM1; i++)
    {
        fc1_bias_buf[i] = *fc1_bias++;
    }
    Loop_rd2:for (int i = 0; i < FC2_OUTPUT_NUM1; i++)
    {
        fc2_bias_buf[i] = *fc2_bias++;
    }    
    Loop_rd3:for (int i = 0; i < FC3_OUTPUT_NUM1; i++)
    {
        fc3_bias_buf[i] = *fc3_bias++;
    }


        Loop_rd4:for (int n = 0; n < CONV_K; n++)
        {
				Loop_rd5:for (int i = 0; i < CONV1_CHout; i++)
				{
            		conv1_W_buf[i][0][0][n] = *conv1_w++;
				}
        }
    
    Loop_rd6:for (int i = 0; i < CONV1_CHout; i++)
	{
		conv1_bias_buf[i] = *conv1_bias++;
	}
	Loop_rd7:for (int i = 0; i < CONV2_CHout; i++)
	{
		conv2_bias_buf[i] = *conv2_bias++;
	}
	Loop_rd8:for (int i = 0; i < CONV3_CHout; i++)
	{
		conv3_bias_buf[i] = *conv3_bias++;
	}	

    Loop_rd9:for (int i = 0; i < input_R; i++)
    {
        Loop_rd10:for (int j = 0; j < input_C; j++)
        {
            Loop_rd11:for (int chi = 0; chi < input_channel; chi++)
            {
            	if(j < (input_C-2))
				{
                	input_data_buf[chi][i][j] = *input_data++;
                }
                else
                {
                	input_data_buf[chi][i][j] = 0.0;
				}
            }            
        }        
    }
    /****************************************
     ********* CONV1_LAYER**************
     *********************************/

        Loop_cv1:for (int i = 0; i < CONV1_CHout; i++)
		{
			Loop_cv2:for (int j = 0; j < CONV1_R; j++)
			{
				Loop_cv3:for (int m = 0; m < CONV1_C; m++)
				{
					conv1_out[i][j][m] = 0;
				}
			}
		}


		Loop_cv4:for(int kc=0; kc<CONV_K; kc++)  //3
		{
				Loop_cv5:for(int c=0; c<CONV1_C; c++)  //258
				{
					Loop_cv6:for(int cho=0; cho<CONV1_CHout; cho++)  //32
					{
							#pragma HLS UNROLL factor=8
							conv1_out[cho][0][c] += input_data_buf[0][0][c+kc] * conv1_W_buf[cho][0][0][kc];						
					}
				}
		}


		Loop_cv7:for (int i = 0; i < CONV1_CHout; i++)
		{			
				Loop_cv8:for (int m = 0; m < CONV1_C; m++)
				{
					conv1_out[i][0][m] += conv1_bias_buf[i];
					conv1_out[i][0][m] = (conv1_out[i][0][m] > 0) ? conv1_out[i][0][m] : (d16)0;
					//printf("%f \t",conv1_out[i][j][m]);
				}
				//printf("\n");
			//printf("\n");
		}
    /****************************************
     ********* POOLING1_LAYER**************
     *********************************/
        Loop_POL1:for (int chi = 0; chi < POOLING1_CHin; chi++)
        {
                Loop_POL2:for (int c = 0; c < POOLING1_C+2; c++)
                {
                	if(c < POOLING1_C)
                	{
                    	d16 max = conv1_out[chi][0][2*c];
                        	Loop_POL3:for (int kc = 0; kc < POOLING_K; kc++)
                        	{
                            	max = (conv1_out[chi][0][2*c+kc] > max) ? conv1_out[chi][0][2*c+kc] : max;
                        	}
						pooling1_out[chi][0][c] = max;
                    }
                    else
                    {
                    	pooling1_out[chi][0][c] = 0.0;
                    }
                }
        }

    /****************************************
     ********* CONV2_LAYER**************
     *********************************/
        Loop2_cv1:for (int i = 0; i < CONV2_CHout; i++)
		{
				Loop2_cv2:for (int m = 0; m < CONV2_C; m++)
				{
					conv2_out[i][0][m] = 0;
				}
		}


		    Loop2_cv3:for (int kc = 0; kc < CONV_K; kc++)
		    {
                Loop2_cv4:for (int i = 0; i < CONV2_CHin; i++)
                {
                    Loop2_cv5:for (int j = 0; j < CONV2_CHout; j++)
                    {
						//#pragma HLS UNROLL factor=8
                        conv2_W_buf[j][i] = *conv2_w++;
                    }
                }


                    Loop2_cv6:for(int c=0; c<CONV2_C; c++) //129
                    {
                        Loop2_cv7:for(int chi=0; chi<CONV2_CHin; chi++) //32
                        {
                            Loop2_cv8:for(int cho=0; cho<CONV2_CHout; cho++) //64
                            {
								#pragma HLS UNROLL factor=8
                                conv2_out[cho][0][c] += pooling1_out[chi][0][c+kc] * conv2_W_buf[cho][chi];
                            }
                        }
                    }                
            }
        Loop2_cv9:for (int i = 0; i < CONV2_CHout; i++)
		{
				Loop2_cv10:for (int m = 0; m < CONV2_C; m++)
				{
					conv2_out[i][0][m] += conv2_bias_buf[i];
					conv2_out[i][0][m] = (conv2_out[i][0][m]>0) ? conv2_out[i][0][m] : (d16)0;
				}
		}



    /****************************************
     ********* POOLING2_LAYER**************
     *********************************/
        Loop2_POL1:for (int chi = 0; chi < POOLING2_CHin; chi++)
        {
                Loop2_POL2:for (int c = 0; c < POOLING2_C+2; c++)
                {
                	if(c < POOLING2_C)
                	{                	
                    	d16 max = conv2_out[chi][0][2*c];
                        	Loop2_POL3:for (int kc = 0; kc < POOLING_K; kc++)
                        	{
                            	max = (conv2_out[chi][0][2*c+kc] > max) ? conv2_out[chi][0][2*c+kc] : max;
                        	}
                    	pooling2_out[chi][0][c] = max;
                    }
                    else
                    {
                    	pooling2_out[chi][0][c] = 0.0;
					}
                }
        }

    /****************************************
     ********* CONV3_LAYER**************
     *********************************/
        Loop3_cv1:for (int i = 0; i < CONV3_CHout; i++)
		{
				Loop3_cv2:for (int m = 0; m < CONV3_C; m++)
				{
					conv3_out[i][0][m] = 0;
				}
		}


		    Loop3_cv3:for (int kc = 0; kc < CONV_K; kc++) //3
		    {
                Loop3_cv4:for (int i = 0; i < CONV3_CHin; i++) //64
                {
                    Loop3_cv5:for (int j = 0; j < CONV3_CHout; j++) //64
                    {
						//#pragma HLS UNROLL factor=8
                        conv3_W_buf[j][i] = *conv3_w++;
                    }
                }

                    Loop3_cv6:for(int c=0; c<CONV3_C; c++) //64
                    {
                        Loop3_cv7:for(int chi=0; chi<CONV3_CHin; chi++) //64
                        {
                            Loop3_cv8:for(int cho=0; cho<CONV3_CHout; cho++)  //64
                            {
								#pragma HLS UNROLL factor=8
                                conv3_out[cho][0][c] += pooling2_out[chi][0][c+kc] * conv3_W_buf[cho][chi];
                            }
                        }
                    }
            }
        Loop3_cv9:for (int i = 0; i < CONV3_CHout; i++)
		{
				Loop3_cv10:for (int m = 0; m < CONV3_C; m++)
				{
					#pragma HLS PIPELINE
					conv3_out[i][0][m] += conv3_bias_buf[i];
					conv3_out[i][0][m] = (conv3_out[i][0][m]>0) ? conv3_out[i][0][m] : (d16)0;
				}
		}

    /****************************************
     ********* POOLING3_LAYER**************
     *********************************/        
        Loop3_POL1:for (int chi = 0; chi < POOLING3_CHin; chi++)
        {
                Loop3_POL2:for (int c = 0; c < POOLING3_C; c++)
                {
                    	d16 max = conv3_out[chi][0][2*c];	
                        	Loop3_POL3:for (int kc = 0; kc < POOLING_K; kc++)
                        	{
                            	max = (conv3_out[chi][0][2*c+kc] > max) ? conv3_out[chi][0][2*c+kc] : max;
                        	}
                    	pooling3_out_buf[chi][0][c] = max;
                }
        }
//  flattening layer
        int cnt = 0;

            Loop3_POL4:for (int c = 0; c < POOLING3_C; c++)
            {
                Loop3_POL5:for (int cho = 0; cho < POOLING3_CHout; cho++)
                {
                    #pragma HLS PIPELINE
                    pooling3_out[cnt] = pooling3_out_buf[cho][0][c];
                    //printf("%f \n",pooling3_out[cnt]);
                    cnt++;
                }
            }


    /****************************************
     ********* FULL_CONNECTION1_LAYER**************
     *********************************/
                    Loop1_fc1:for (int cho = 0; cho < FC1_OUTPUT_NUM1; cho++)
                    {
                        fc1_out[cho] = 0;
                    }

	                Loop1_fc2:for (int cho = 0; cho < FC2_OUTPUT_NUM1; cho++)
    	            {
        	            fc2_out[cho] = 0;
            	    }
    
					Loop1_fc4:for(int x=0;x<6;x++)
					{
					fc1_outu[x]	 = 0;
					fc2_outu[x] = 0;
					}  					       

        Loop1_fc5:for (int i = 0; i < FC1_INPUT_NUM1; i++)
        {      
        	Loop1_fc6:for(int j=0;j<6;j++)
        	{
        			#pragma HLS PIPELINE
        			d16 fc1ua_w_val = (*(fc1u_w + (i*6 + j)));       	
                    fc1_outu[j] += pooling3_out[i] * fc1ua_w_val;
            }
        }
        
        Loop1_fc7:for (int i = 0; i < 6; i++)
        {
                 Loop1_fc8:for (int j = 0; j < 512; j++)
                {

        			#pragma HLS PIPELINE
        			d16 fc1v_w_val = (*(fc1v_w + (i*512+j)));
                    fc1_out[j] += fc1_outu[i] * fc1v_w_val;
                }
        }        
        
        Loop1_fc9:for (int i = 0; i < FC1_OUTPUT_NUM1; i++)
        {
            fc1_out[i] += fc1_bias_buf[i];
            fc1_out[i] = (fc1_out[i] > 0) ? fc1_out[i] : (d16)0;
        }     
    /****************************************
     ********* FULL_CONNECTION2_LAYER**************
     *********************************/     
        Loop2_fc1:for (int i = 0; i < FC2_INPUT_NUM1; i++)
        {
        	Loop2_fc2:for(int j=0;j<6;j++)
        	{
        			#pragma HLS PIPELINE
        			d16 fc2ua_w_val = (*(fc2u_w + (i*6 + j)));              	
                	fc2_outu[j] += fc1_out[i] * fc2ua_w_val;
			}
        }    

		Loop2_fc3:for (int i = 0; i < 6; i++)
        {   	
            Loop2_fc4:for (int j = 0; j < FC2_OUTPUT_NUM1; j++)
            {
        			#pragma HLS PIPELINE
        			d16 fc2v_w_val = (*(fc2v_w + (i*64+j)));
                fc2_out[j] += fc2_outu[i] * fc2v_w_val;
            }
        }

        Loop2_fc5:for (int i = 0; i < FC2_OUTPUT_NUM1; i++)
        {
            fc2_out[i] += fc2_bias_buf[i];
            fc2_out[i] = (fc2_out[i] > 0) ? fc2_out[i] : (d16)0;
           // printf("%f \n",(d16)fc2_out[i]);
        }     
     
  /****************************************
     ********* FULL_CONNECTION3_LAYER**************
     *********************************/
        Loop3_fc1:for (int i = 0; i < FC3_INPUT_NUM1; i++)
        {
            Loop3_fc2:for (int j = 0; j < FC3_OUTPUT_NUM1; j++)
            {
        		#pragma HLS PIPELINE
                fc3_weights_buf[j] = *fc3_w++;
            }
            if(i == 0)
            {
                Loop3_fc3:for (int cho = 0; cho < FC3_OUTPUT_NUM1; cho++)
                {
                    output_data_buf[cho] = 0;
                }
            }
            Loop3_fc4:for (int j = 0; j < FC3_OUTPUT_NUM1; j++)
            {
    			#pragma HLS PIPELINE
                output_data_buf[j] += fc2_out[i] * fc3_weights_buf[j];
            }
        }

        Loop3_fc5:for (int i = 0; i < FC3_OUTPUT_NUM1; i++)
        {
            *output_data= fc3_bias_buf[i];
        }
   // return;
}
     
     
     
     
     
     
     
    
