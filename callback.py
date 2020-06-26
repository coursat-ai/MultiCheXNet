class CustomCallBacks():
    """
    How to Use:
    monitor = CustomeCallBacksObject.calls('DetectionModel.h5',
                  checkpointMonitor = 'val_loss',
                  checkpointMode = 'min',
                  earlyStopMonitor = 'val_loss',
                  earlyStopPatience = 10,
                  earlyStopMode = 'auto',
                  useReduceOnPlateau = False,
                  useLearningRateScheduler = True)
                  
    model.fit(..., callbacks=monitor)
    """
    
    def calls(self,
              model_name, 
              checkpointMonitor = 'val_loss',
              checkpointMode = 'min',
              earlyStopMonitor = 'val_loss',
              earlyStopPatience = 10,
              earlyStopMode = 'auto',
              useReduceOnPlateau = False,
              useLearningRateScheduler = False
              ):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                                    monitor= checkpointMonitor,
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode=checkpointMode,
                                    period=1)
        
        early = tf.keras.callbacks.EarlyStopping(monitor=earlyStopMonitor,
                                min_delta=0,
                                patience=earlyStopPatience,
                                verbose=1,
                                mode=earlyStopMode)
        
        class myCallBack(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('val_accuracy') > 0.998):
                    print ('\nReached 0.998 Validation accuracy!')
                    self.model.stop_training = True

        my_call = myCallBack()

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5,
                                    verbose=1, mode='min', min_delta=0,
                                    cooldown=0, min_lr=0)
        
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 10))
        
        monitor = [checkpoint, early]
        if useReduceOnPlateau:
            monitor.append(reduce_lr)
        if useLearningRateScheduler:
            monitor.append(lr_schedule)
        return monitor
