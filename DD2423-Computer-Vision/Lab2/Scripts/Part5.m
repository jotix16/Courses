%% Test
t = triangle128;
c = zerocrosscurves(t-128);
houghline(1,c,t, 1000, 300, 10, 3, 2);

%% Best hough for triangle
testimage1 = triangle128;
smalltest1 = binsubsample(testimage1);
houghedgeline(1,testimage1, 4, 100, 1400, 400, 3, 2);


%% Best hough for trapezoid and triangle
testimage2 = houghtest256;
houghedgeline(1,testimage2, 5, 100, 4080, 180, 7, 1);

%% Best hough for tools
tools = few256;
houghedgeline(1,tools, 3, 4000, 4500, 150, 10, 1);

%% Best hough for phone
phone = phonecalc256;
houghedgeline(1,phone, 7, 10600, 6500, 130, 10, 1);


%% Best hough for house
house = godthem256;
houghedgeline(1,house, 12, 450, 4000, 170, 7, 1);
