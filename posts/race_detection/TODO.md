TODO Tasks

Algorithm
* Everyone should read the papers https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext
* Plot things that could distinguish - number of images/revisits by race, number of lateral/non-lateral images, age by race, gender in each race, etc., anything that could identify patients
* Still need analysis on all different datasets - look at things like number of patients in the normal dataset, bwa subset, bwa_equal subset, etc.
* Use pretrained EfficientNet and maybe do a self-implementation
* Optimizer
* * completed hold out validation for learning rate
* * I've hard coded the model into the optimization function, this should be an input though - issue is for each parameter model has to reset
* * try learning rate schedules (which will change learning rate over time)
* * optimize other hyperparamters - batch size, number of epochs, weights for loss function, different loss functions (than cross entropy loss)ion for loss
* Add weight function for loss so even with uneven dataset we can achieve better accuracy on all populations
* Image augmentation - don't have blurs or zooms, this could be good to add blurs/zooms
* Analysis wise, could be interesting to do intersectional bias - black women, etc.

Ethics
* Find papers - would be helpful after each paper to put small summary on what paper is about, major findings

Finished
* Implemented DataLoader
* Most of data in shared drive - AND check that data is actually loaded
* By the paper, "White" is anything containing label White - "White", "White, non-Hispanic", "White Hispanic"
* Decided to do lateral and frontal image scans
* Implemented ResNet34 via own implementation
* Made df_patients_race that have patient image link, patient id, patient race, patient race num (based on White, Black, Asian, Other)
* finished hold-out optimization loop for learning rate
* Added image augmentation
