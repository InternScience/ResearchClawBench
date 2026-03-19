******************************************************************************************
Oxford Battery Degradation Dataset 1
******************************************************************************************
The Oxford Battery Degradation Dataset 1 contains measurements of battery ageing data from 8 small lithium-ion pouch cells. The cells were all tested in a thermal chamber at 40 degC. The cells were exposed to a constant-current-constant-voltage charging profile, followed by a drive cycle discharging profile that was obtained from the urban Artemis profile. Characterisation measurements were taken every 100 cycles.

There are two data files given within the dataset:
1. 'ExampleDC_C1.mat' - contains recorded voltage, current and temperature data from the first charge-discharge cycle only, and
2. 'Oxford_Battery_Degradation_Dataset_1.mat' - contains the results of the characterisation tests that were taken every 100 cycles until cell end of life. !! NOTE: the file size of this file is 262 Mb. !!

Further details on the two files are provided below.

---- PLEASE READ THE FOLLOWING MESSAGE CAREFULLY BEFORE PROCEEDING FURTHER:
If you make use of our data, please cite our dataset directly using its DOI, as well as the following PhD thesis: Christoph R. Birkl, "Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries", PhD thesis, Department of Engineering Science, University of Oxford, 2017. 

Please also consider citing our papers on these topics, see:
https://scholar.google.co.uk/citations?hl=en&user=AZdBXIkAAAAJ&view_op=list_works&sortby=pubdate

The data are stored in ".mat" files, which is the Matlab binary file format. You need Matlab, available from Mathworks, to open this type of file.

Thanks for your interest in our work.
David Howey 
Christoph Birkl
June 2017

******************************************************************************************
Description of data contained in 'Oxford_Battery_Degradation_Dataset_1.mat'
******************************************************************************************
Recorded by: Christoph R. Birkl, christoph.birkl@eng.ox.ac.uk
Supervisor: David A. Howey, david.howey@eng.ox.ac.uk

Place of tests: University of Oxford, Howey Research Group
Website: epg.eng.ox.c.uk/howey
Start date of tests: 08/01/2015
Test subjects: 8 x Kokam CO LTD, SLPB533459H4, 740mAh
Battery tester: Bio-Logic MPG-205, 8 channel
Environmental chamber: Binder thermal chamber, environmental temperature: 40 degree Celsius

Data type Matlab binary file containing:
(a) 1-C cycles (current = 740mA): time (t, in seconds), voltage (v, in Volts), charge (q, in mAh) and temperature (T, in degrees Celsius),
(b) Pseudo-OCV cycles (current = 40mA): time (t, in seconds), voltage (v, in Volt), charge (q, in mAh) and temperature (T, in degree Celsius),
1-C and pseudo-OCV cycles were recorded every 100 cycles of drive cycles. 
The drive cycles were based on an Urban Artemis driving profile.

File structure: 
Layer 1: Cells (1-8)
Layer 2: Cycle number of characterisation cycles (e.g. cyc0100: characterisation cycles after 100 drive cycles)
Layer 3: 1-C charge (C1ch), 1-C discharge (C1dc), pseudo-OCV charge (OCVch), pseudo-OCV discharge (OCVdc) 
Layer 4: time (t), voltage (v), charge (q), temperature (T)

******************************************************************************************
Description of data contained in 'ExampleDC_C1.mat'
******************************************************************************************
Same test details as above, but contains only the first charge-discharge cycle data to give an indication of the load profile.

Data type Matlab binary file containing:
(a) A single (first cycle) 2-C constant current charge (current i = 1480 mA): time (t, in seconds), voltage (v, in Volts), charge (q, in mAh) and temperature (T, in degrees Celsius), 
(b) A single (first cycle) variable current discharge, using a drive cycle derived from the Artemis Urban profile: time (t, in seconds), voltage (v, in Volts), current (i, in mA), charge (q, in mAh) and temperature (T, in degrees Celsius).

File structure: 
Layer 1: Charge or Discharge (.ch and .dc)
Layer 2: Time (t), voltage (v), charge (q), temperature (T), current (i)

******************************************************************************************
These data are copyright (c) 2017, The Chancellor, Masters and Scholars of the University of Oxford, and the 'Oxford Battery Degradation Dataset 1' researchers. All rights reserved.

This 'Oxford Battery Degradation Dataset 1' is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/
		
THIS DATA IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


