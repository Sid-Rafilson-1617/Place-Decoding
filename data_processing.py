import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat
import concurrent.futures
import os


#___________Helper functions____________________________

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def cv_split(data, k, k_CV=2, n_blocks=10):
    '''
    Perform an 80/20 cross-validation split of the data, following the Hardcastle et 
    al paper.
    
    Parameters
    --
    data : An array of data.
    
    k : Which CV subset to hold out as testing data (integer from 0 to k_CV-1).
    
    k_CV : Number of CV splits (integer).
        
    n_blocks : Number of blocks for initially partitioning the data. The testing
        data will consist of a fraction 1/k_CV of the data from each of these
        blocks.
        
    Returns
    --
    data_train, data_test : Data arrays after performing the train/test split.
    '''

    block_size = len(data)//n_blocks
    mask_test = [False for _ in data]
    for block in range(n_blocks):
        i_start = int((block + k/k_CV)*block_size)
        i_stop = int(i_start + block_size//k_CV)
        mask_test[i_start:i_stop] = [True for _ in range(block_size//k_CV)]
    mask_train = [not a for a in mask_test]
    data_test = data[mask_test]
    data_train = data[mask_train]

    return data_train, data_test


def preprocess_data(path, bin_size=20, crop_hf=True, flip_data=None, make_plots=False):
    '''
    Get the data from a given file path and preprocess by binning spikes,
    tracking, and sniffing data.
    
    Some preprocessing is also done to exclude 
    units with a large number of refractory violations or negative spike 
    amplitudes (see the readme file).
    
    Parameters
    --
    path : Path to the directory containing the data.
    
    bin_size : Number of 10ms time steps to include in each bin.
    
    crop_hf : If True, crop out the initial and final periods of the 
        session during which the mouse is head fixed.
        
    make_plots : If True, make some plots of the data, saving them as a
        PDF file in the directory containing the data.
        
    flip_data : If flip_data='pre' and there is a floor flip, return 
        only data from before the flip. If flip_data='post' and there 
        is a floor flip, return only data from after the flip.
        
    Returns
    --
    If there is no tracking data for the session, returns None. If there
    is tracking data, the following are returned:
    
    spks : The binned spiking data. 2D array of shape n_bins-by-n_units.
    
    pos_ss : The average x-y position of the mouse's head in each time bin.
        2D array of shape n_bins-by-2.
        
    speed_ss : The average speed (in arbitrary units) of the mouse in each
        time bin. 1D array of length n_bins.
        
    sniff_freq_ss : The average instantaneous sniff frequency (in Hz) in
        each time bin. 1D array of length n_bins. If sniff data is not
        available for the session, returns None.
    '''


    if 'track.mat' in os.listdir(path):
        track = loadmat(path + '/track.mat')
        head = track['track'][:,3:5]
        frame_times_ds = track['track'][:,0]  # frame times in ms (one frame per ~10ms)
    elif 'mtrack.mat' in os.listdir(path):
        track = loadmat(path + '/mtrack.mat')
        head = track['mtrack'][:,2:4]
        frame_times_ds = track['mtrack'][:,0] # frame times in ms (one frame per ~10ms)
    else:
        print('No tracking data for this session.')
        return None, None, None, None

    
    x_max, y_max = 1200, 500  # scale the x and y position data to fall within this range

    cluster_ids = loadmat(path + '/cluster_ids.mat')
    spike_times = loadmat(path + '/spike_times.mat')

    if 'microvolts.mat' in os.listdir(path):
        microvolts = loadmat(path + '/microvolts.mat')
    else:
        print('No microvolt data for this session.')
        return None, None, None, None

    # !!!!!!!!!!!!!!! These files dont exist in the rnp_file I have. Using a temp fix until we figure it out !!!!!!!!!!!!!!
    #frames = loadmat(path + '/gpio_locs.mat')
    #frame_times = frames['gpio_locs'][0]  # frame times according to ticks of a 30kHz clock (one frame per ~300 steps)
    frame_times = frame_times_ds * 30  # convert to 30kHz clock

    spike_key = spike_times.keys()
    spike_key = list(spike_key)[-1]
    spikes = spike_times[spike_key][:,0]  # spike times according to ticks of a 30kHz clock
    clusters = cluster_ids['clusters'][:,0]
    mv = microvolts['master_mv'][:,0]
    
    



    pos = head
    speed = (pos[1:,:] - pos[:-1,:])/np.outer((frame_times_ds[1:] - frame_times_ds[:-1]), np.ones(2))
    speed = np.vstack((speed, np.zeros(2).T))
    
    # Occasionally the shapes of the following two things differ slightly, so chop one:
    if len(frame_times) != len(frame_times_ds):
        print('frame_times and frame_times_ds have different sizes: ', 
              len(frame_times), len(frame_times_ds))
        min_len = np.min([len(frame_times), len(frame_times_ds)])
        frame_times = frame_times[:min_len]
        frame_times_ds = frame_times_ds[:min_len]
        pos = pos[:min_len]
        speed = speed[:min_len]
    n_frames = len(frame_times_ds)

    # Interpolate nans:
    for i in range(2):
        nans, x = nan_helper(pos[:,i])
        if np.sum(nans) > 0:
            pos[nans,i]= np.interp(x(nans), x(~nans), pos[~nans,i])
        nans, x = nan_helper(speed[:,i])
        if np.sum(nans) > 0:
            speed[nans,i]= np.interp(x(nans), x(~nans), speed[~nans,i])

    # Preprocess the sniff data (if it exists):
    if 'sniff_params.mat' in os.listdir(path): 
        sniff = loadmat(path + '/sniff_params.mat')['sniff_params']  # sniff times in ms
        sniffs = sniff[:,0]
        #bad sniffs are sniffs where the third column is zero
        bad_sniffs = np.where(sniff[:,2] == 0)[0]

        sniffs = np.delete(sniffs, bad_sniffs)

        dsniffs = sniffs[1:] - sniffs[:-1]
        sniffs = sniffs[1:]
        sniff_freq = 1000/dsniffs  # instantaneous sniff frequency (in Hz)
        sniff_freq_binned = np.zeros(n_frames)
        for i,t in enumerate(frame_times_ds):
            sniff_freq_binned[i] = np.mean(sniff_freq[(sniffs>t)*(sniffs<t+10*bin_size)])

        # Interpolate nans (in case some bins didn't have sniffs):
        nans, x = nan_helper(sniff_freq_binned)
        if np.sum(nans) > 0:
            sniff_freq_binned[nans]= np.interp(x(nans), x(~nans), sniff_freq_binned[~nans])
    else:
        print('No sniff data for this session.')
        sniff_freq_binned = None



    if 'events.mat' in os.listdir(path): 
        events = loadmat(path + '/events.mat')['events']

        # Event frames and times:
        frame_fm1, t_fm1 = events[0,0], events[0,2]  # frame/time at which initial HF condition ends
        frame_fm2, t_fm2 = events[0,1], events[0,3]  # frame/time at which FM condition begins
        frame_hf1, t_hf1 = events[1,0], events[1,2]  # frame/time at which FM condition ends
        frame_hf2, t_hf2 = events[1,1], events[1,3]  # frame/time at which final HF condition begins
        frame_flip1, t_flip1 = events[2,0], events[2,2]  # frame/time at which floor flip begins
        frame_flip2, t_flip2 = events[2,1], events[2,3]  # frame/time at which floor flip ends

        # Create a mask to handle head-fixed to freely moving transitions and floor flips:
        mask = np.array([True for ii in range(n_frames)])
        color_mask = np.array([0 for ii in range(n_frames)]) # for plotting purposes. 0 for free movement, 1 for headfixed, and 2 for transitions
        if crop_hf:  # crop out the initial and final HF periods
            if frame_fm1!=0:
                mask[:frame_fm2] = False
                color_mask[:frame_fm2] = 1
            if frame_hf1!=0:
                mask[frame_hf1:] = False
                color_mask[frame_hf1:] = 1
        else:  # crop out just the transitions between FM and HF
            if frame_fm1!=0:
                mask[frame_fm1:frame_fm2] = False
                color_mask[frame_fm1:frame_fm2] = 2
            if frame_hf1!=0:
                mask[frame_hf1:frame_hf2] = False
                color_mask[frame_hf1:frame_hf2] = 2
        if frame_flip1!=0:  # keep data only from before or after the flip
            mask[frame_flip1:frame_flip2] = False
            if flip_data=='pre':  
                mask = np.array([f < frame_flip1 for f in range(n_frames)])*mask
            elif flip_data=='post':
                mask = np.array([f > frame_flip2 for f in range(n_frames)])*mask
            #elif flip_data is None:
                #mask = np.array([f < frame_flip1 or f > frame_flip2 for f in range(n_frames)])*mask
            



            if False: ### I never used this !!!!
                spiketimes = []
                if frame_flip1 < len(frame_times) and frame_flip2 < len(frame_times):
                    for t_s in spikes:
                        if t_s < frame_times[frame_flip1] or t_s > frame_times[frame_flip2]:
                            spiketimes.append(t_s)
                    spikes = np.array(spike_times)
                else:
                    print(f"Warning: frame_flip1 ({frame_flip1}) or frame_flip2 ({frame_flip2}) is out of bounds for frame_times with size {len(frame_times)}")
                    return None, None, None, None


            # ensuring mask in long enough
            if np.sum(mask) < 10_000:
                print('Mask is small or empty after applying floor flip conditions.')
                return None, None, None, None
            
        # plot the sniff frequencies color coded by the 3 conditions
        if sniff_freq_binned is not None and make_plots:
            plt.figure(figsize=(20,8))
            plt.scatter(frame_times_ds[color_mask==0], sniff_freq_binned[color_mask==0], s=5, marker='.')
            plt.scatter(frame_times_ds[color_mask==1], sniff_freq_binned[color_mask==1], s=5, marker='.')
            plt.scatter(frame_times_ds[color_mask==2], sniff_freq_binned[color_mask==2], s=5, marker='.')

            # draw two vertical lines to indicate the floor flip
            if t_flip1 != 0 and t_flip2 != 0:
                plt.axvline(x=t_flip1, color='k', linestyle='--')
                plt.axvline(x=t_flip2, color='k', linestyle='--')
                plt.title(f'Sniff frequency color coded by condition\nFloor flip times {t_flip1, t_flip2}')
            else:
                plt.title('Sniff frequency color coded by condition\nNo floor flip')
            plt.xlabel('Time (ms)')
            plt.ylabel('Sniff frequency (Hz)') 
            plt.legend(['Free movement', 'Head fixed', 'Transitions', 'Floor flip'])
            plt.tight_layout()
            plt.savefig(path + '/sniff_frequency_color_coded.png')
            plt.close()



        # Keep the data selected by the mask; 
        frame_times_ds = frame_times_ds[mask]
        frame_times = frame_times[mask]
        pos = pos[mask,:]
        speed = speed[mask,:]

        # Chop off the last few points if not divisible by bin_size:
        frame_times_ds = frame_times_ds[:bin_size*(len(frame_times_ds)//bin_size)]
        frame_times = frame_times[:bin_size*(len(frame_times)//bin_size)]
        pos = pos[:bin_size*(len(pos)//bin_size)]
        speed = speed[:bin_size*(len(speed)//bin_size)]

        # Do the same thing for the sniff data if it exists:
        if 'sniff_params' in os.listdir(path): 
            sniff_freq_binned = sniff_freq_binned[mask]
            sniff_freq_binned = sniff_freq_binned[:bin_size*(len(sniff_freq_binned)//bin_size)]
        
            # Average the sniff-frequency data within each bin:
            sniff_freq_ss = np.zeros(len(sniff_freq_binned)//bin_size)
            for i in range(len(sniff_freq_binned)//bin_size):
                sniff_freq_ss[i] = np.mean(sniff_freq_binned[i*bin_size:(i+1)*bin_size], axis=0)
        else:
            sniff_freq_ss = None

    # Average the behavioral data within each bin:
    pos_ss = np.zeros((len(pos)//bin_size, 2))
    speed_ss = np.zeros((len(speed)//bin_size, 2))
    for i in range(len(pos)//bin_size):
        pos_ss[i,:] = np.mean(pos[i*bin_size:(i+1)*bin_size,:], axis=0)
        speed_ss[i,:] = np.mean(speed[i*bin_size:(i+1)*bin_size,:], axis=0)

    # Clip and normalize the position data:
    pos_ss[:,0] = np.clip(pos_ss[:,0], np.percentile(pos_ss[:,0], 0.5), np.percentile(pos_ss[:,0], 99.5))
    pos_ss[:,1] = np.clip(pos_ss[:,1], np.percentile(pos_ss[:,1], 0.5), np.percentile(pos_ss[:,1], 99.5))
    pos_ss[:,0] -= np.min(pos_ss[:,0])
    pos_ss[:,1] -= np.min(pos_ss[:,1])
    pos_ss[:,0] = pos_ss[:,0]*x_max/np.max(pos_ss[:,0])
    pos_ss[:,1] = pos_ss[:,1]*y_max/np.max(pos_ss[:,1])
        
    # Bin the spiking data:
    spks = np.zeros((0, len(frame_times)//bin_size))
    for cluster in np.unique(clusters):
        # only keep clusters with firing rate > 0.5 Hz:
        c1 = np.sum(spikes[clusters==cluster])/(1e-3*(frame_times[-1] - frame_times[0])) > 0.5

        # < 5% of spikes may violate the 1.5ms refractory period:
        isi = np.diff(spikes[clusters==cluster])
        c2 = np.sum(isi < 1.5)/(1+len(isi)) < 0.05  

        # < 10% of spikes may have negative amplitude:
        mv[np.isnan(mv)] = -1  # nans count the same as negative amplitudes
        mv_cluster = mv[clusters==cluster]
        mv_nonzero = mv_cluster[mv_cluster!=0]
        c3 = np.sum(mv_nonzero<0)/(1+len(mv_nonzero)) < 0.1

        if c1 and c2 and c3:
            bin_edges = np.append(frame_times[::bin_size], frame_times[-1])
            spike_counts, _ = np.histogram(spikes[clusters==cluster], bin_edges, density=False)
            # Normalize so that spike counts are in Hz:
            spike_counts = 3e4*spike_counts/(bin_edges[1:] - bin_edges[:-1])
            spks = np.vstack((spks, spike_counts[:len(spks.T)])) 
    spks = spks.T

    if make_plots:
        times = np.arange(len(pos_ss))*0.01*bin_size
        plt.figure(figsize=(9,9))
        plt.subplot(421)
        plt.plot(times, pos_ss[:,0])
        plt.plot(times, pos_ss[:,1])
        plt.xlim(0, times[-1])
        plt.ylabel('x,y')
        plt.xlabel('Time (s)')
        plt.subplot(422)
        plt.plot(pos_ss[:,0], pos_ss[:,1], lw=0.25)
        plt.xlim(0,x_max)
        plt.ylim(0,y_max)
        plt.subplot(423)
        plt.plot(times, np.linalg.norm(speed_ss, axis=1))
        plt.xlim(0, times[-1])
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (a.u.)')
        plt.subplot(424)
        plt.hist(100/bin_size*np.mean(spks, axis=0), bins=30)
        plt.xlabel('Firing rate (Hz)')
        plt.ylabel('Units')
        if sniff_freq_ss is not None:
            plt.subplot(425)
            plt.plot(times, sniff_freq_ss)
            plt.xlim(0, times[-1])
            plt.xlabel('Time (s)')
            plt.ylabel('Sniff frequency (Hz)')
        plt.subplot(427)
        plt.imshow(np.log(1e-3+(spks/np.max(spks, axis=0)).T), aspect='auto', interpolation='none')
        plt.xticks([0, len(spks)], [0, int(times[-1])])
        plt.yticks([0, len(spks.T)-1], [1, len(spks.T)])
        plt.xlabel('Time (s)')
        plt.ylabel('Unit')
        plt.tight_layout()
        if flip_data is not None:
            plt.savefig(path + '/data_FNO_F' + flip_data +'.pdf')
        else:
            plt.savefig(path + '/data_FNO_F.pdf')

    return spks, pos_ss, speed_ss, sniff_freq_ss





#___________Decoding with floor flip______________________

def process_fold_floorflip(kk, data_pre, data_post, data_spks_pre, data_spks_post, bin_numbers_pre, bin_numbers_post, n_squares, boundaries, bins_nodata, kCV, train_on, test_on, rotate):

    """
    Processes a single cross-validation fold for spatial decoding with a floor flip event, 
    using spike data to predict positions within specified spatial boundaries. This function
    performs batch decoding to improve computational efficiency.

    Parameters:
    -----------
    kk : int
        The current cross-validation fold number.
    data_pre : numpy.ndarray
        The 2D array containing positional data before the floor flip (shape: [timesteps, 2]).
    data_post : numpy.ndarray
        The 2D array containing positional data after the floor flip (shape: [timesteps, 2]).
    data_spks_pre : numpy.ndarray
        The 2D array of spike count vectors before the floor flip (shape: [timesteps, features]).
    data_spks_post : numpy.ndarray
        The 2D array of spike count vectors after the floor flip (shape: [timesteps, features]).
    bin_numbers_pre : numpy.ndarray
        Array of spatial bin assignments for each pre-floor flip timestep.
    bin_numbers_post : numpy.ndarray
        Array of spatial bin assignments for each post-floor flip timestep.
    n_squares : tuple of int
        Tuple representing the number of bins in the x and y directions, respectively (nx, ny).
    boundaries : tuple of float
        Tuple representing the spatial boundaries (xmin, xmax, ymin, ymax).
    bins_nodata : list
        List of bins (indices) for which no data is available, to exclude from training.
    kCV : int
        Total number of cross-validation folds.
    train_on : str
        Specifies the phase of data ('pre' or 'post') to use for training.
    test_on : str
        Specifies the phase of data ('pre' or 'post') to use for testing.
    rotate : bool
        If True, rotates the spatial locations for decoding by 180 degrees.

    Returns:
    --------
    fold_error : float
        The median error across the decoded positions for this fold.
    predicted_positions : numpy.ndarray
        The predicted positions for all timesteps (shape: [n_timesteps, 2]), with positions 
        assigned for test indices and zeros elsewhere.

    Notes:
    ------
    - Trains a linear SVM classifier for each pair of spatial bins to learn decision boundaries 
      between spatial locations.
    - Predicts positions using a batch decoding approach, where spike count vectors are decoded
      in batches to enhance computational efficiency.
    - Accumulates votes for each spatial bin based on the pairwise classifiers, then assigns 
      the bin with the highest votes as the predicted location.
    - Computes error as the Euclidean distance between the predicted and actual positions.
    """
    
    nx, ny = n_squares
    xmin, xmax, ymin, ymax = boundaries
    n_timesteps_pre = len(data_pre)
    n_timesteps_post = len(data_post)

    batch_size = 1000  # Adjust as needed for memory

 
    # Split pre and post data into training and testing sets
    bin_numbers_train_pre, _ = cv_split(bin_numbers_pre, kk, kCV)
    data_spks_train_pre, data_spks_test_pre = cv_split(data_spks_pre, kk, kCV)
    _, data_pos_test_pre = cv_split(data_pre, kk, kCV)
    _, indices_test_pre = cv_split(np.arange(n_timesteps_pre), kk, kCV)

    bin_numbers_train_post, _ = cv_split(bin_numbers_post, kk, kCV)
    data_spks_train_post, data_spks_test_post = cv_split(data_spks_post, kk, kCV)
    _, data_pos_test_post = cv_split(data_post, kk, kCV)
    _, indices_test_post = cv_split(np.arange(n_timesteps_post), kk, kCV)

    # Select training and testing data
    if train_on == 'pre':
        bin_numbers_train = bin_numbers_train_pre
        data_spks_train = data_spks_train_pre
    elif train_on == 'post':
        bin_numbers_train = bin_numbers_train_post
        data_spks_train = data_spks_train_post

    if test_on == 'pre': 
        data_spks_test = data_spks_test_pre
        data_pos_test = data_pos_test_pre
        indices_test = indices_test_pre
    elif test_on == 'post':
        data_spks_test = data_spks_test_post
        data_pos_test = data_pos_test_post
        indices_test = indices_test_post

    # Rotate spatial location if required
    if rotate:
        midline_x = (xmax + xmin) / 2
        midline_y = (ymax + ymin) / 2
        data_pos_test[:, 0] = 2 * midline_x - data_pos_test[:, 0]
        data_pos_test[:, 1] = 2 * midline_y - data_pos_test[:, 1]

    


    # Train classifiers for this fold
    clf_dict = {}
    bins_nodata = []

    def train_pairwise_classifier(i, j):
        bins_i_train = bin_numbers_train[bin_numbers_train == i]
        spks_i_train = data_spks_train[bin_numbers_train == i, :]
        bins_j_train = bin_numbers_train[bin_numbers_train == j]
        spks_j_train = data_spks_train[bin_numbers_train == j, :]

        if len(bins_i_train) == 0:
            return (i, None)
        elif len(bins_j_train) == 0:
            return (j, None)
        else:
            X_train = np.vstack((spks_i_train, spks_j_train))
            y_train = np.concatenate((np.ones_like(bins_i_train), np.zeros_like(bins_j_train)))
            clf = svm.SVC(kernel='linear', class_weight='balanced', C=0.1)
            clf.fit(X_train, y_train)
            return ((i, j), clf)

    # Train classifiers
    for i in range(nx * ny):
        for j in range(i + 1, nx * ny):
            result = train_pairwise_classifier(i, j)
            if result[1] is None:
                bins_nodata.append(result[0])
            else:
                clf_dict[result[0]] = result[1]

    # Decode positions in batches
    def decode_batch(spks_batch, n_locations, clf_dict, bins_nodata):
        batch_size = len(spks_batch)
        batch_votes = np.zeros((batch_size, n_locations))

        for i in range(n_locations):
            for j in range(i + 1, n_locations):
                if (i not in bins_nodata) and (j not in bins_nodata):
                    clf = clf_dict.get((i, j), None)
                    if clf:
                        preds = clf.predict(spks_batch)
                        batch_votes[:, i] += preds
                        batch_votes[:, j] += 1 - preds
        return batch_votes

    # Decode all positions in the test set
    votes = np.zeros((len(data_spks_test), nx * ny))

    for batch_start in range(0, len(data_spks_test), batch_size):
        spks_batch = data_spks_test[batch_start:batch_start + batch_size]
        batch_votes = decode_batch(spks_batch, nx * ny, clf_dict, bins_nodata)
        votes[batch_start:batch_start + batch_size, :] = batch_votes

    # Predicted bins and positions
    bin_pred = np.argmax(votes, axis=1)
    pos_pred = np.zeros((len(bin_pred), 2))
    err = np.zeros(len(bin_pred))

    for i in range(len(bin_pred)):
        pos_pred[i, 0] = xmin + (0.5 + bin_pred[i] % nx) * (xmax - xmin) / nx
        pos_pred[i, 1] = ymin + (0.5 + bin_pred[i] // nx) * (ymax - ymin) / ny
        err[i] = np.linalg.norm(data_pos_test[i, :] - pos_pred[i, :])

    fold_error = np.median(err)
    predicted_positions = np.zeros((n_timesteps_pre if test_on == 'pre' else n_timesteps_post, 2))
    predicted_positions[indices_test, :] = pos_pred

    return fold_error, predicted_positions


def decoding_err_floorflip(data_pre, data_post, data_spks_pre, data_spks_post, n_squares, boundaries, kCV=10, shuffle=False, train_on='pre', test_on='post', rotate = False):
    
    '''
    Train a battery of pairwise classifiers to decode 2D spatial position from spiking
    activity. This function trains on data from 1 floor condition and tests on data from
    the other floor condition. The pre and post data are both split into kCV folds (default 10).
    Instead of testing on the heald out fold from the pre data, the function tests on the corresponding fold from the post data.


    Parameters
    --
    data_pre : Position data giving the 2D position at each observation before the floor flip.
        2D array of size n_observations-by-2.

    data_post : Position data giving the 2D position at each observation after the floor flip.
        2D array of size n_observations-by-2.

    data_spks_pre : Spiking data giving the spike count vector at each observation before the floor flip.
        2D array of size n_observations-by-n_neurons.

    data_spks_post : Spiking data giving the spike count vector at each observation after the floor flip.
        2D array of size n_observations-by-n_neurons.

    n_squares : 2D list giving the number of grid squares along the x and y directions.

    boundaries: A list of four values (xmin, xmax, ymin, ymax) giving the boundaries
        of the arena.

    kCV : An integer specifying how many folds in k-fold cross-validation.

    shuffle : If True, circularly shuffle the neural and behavior data as a control.

    train_on : Specify whether to train on 'pre' or 'post' data.

    test_on : Specify whether to test on 'pre' or 'post' data.

    Returns
    --
    err_kfold : A list of length kCV, where each element corresponds to the median
        decoding error (i.e. distance between the actual and estimated spatial positions)
        from one of the training/testing splits.

    predicted_position : predicted_position[i,:] is the 2D position predicted by the
        decoder in bin i.
    '''


    # getting the 2D shape of the data and checking if they match
    n_timesteps_pre, n_neurons_pre = np.shape(data_spks_pre)
    n_timesteps_post, n_neurons_post = np.shape(data_spks_post)
    if n_neurons_pre != n_neurons_post:
        print('Number of neurons in pre and post data do not match.')
        return None, None, None
    
    n_neurons = n_neurons_pre
    print('n_neurons: ', n_neurons)

   
    # Shuffle the data if required
    if shuffle:
        roll_int_pre = np.random.randint(n_timesteps_pre // 10, n_timesteps_pre - n_timesteps_pre // 10)
        data_pre = np.roll(data_pre[::-1], roll_int_pre)
        roll_int_post = np.random.randint(n_timesteps_post // 10, n_timesteps_post - n_timesteps_post // 10)
        data_post = np.roll(data_post[::-1], roll_int_post)


    # Assign bins to positions
    nx, ny = n_squares
    xmin, xmax, ymin, ymax = boundaries
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    bin_numbers_pre = np.zeros(n_timesteps_pre)
    for i in range(n_timesteps_pre):
        x, y = data_pre[i,:]
        mx = (x-xmin)//dx
        my = (y-ymin)//dy
        bin_numbers_pre[i] = int(nx*my + mx)

    bin_numbers_post = np.zeros(n_timesteps_post)
    for i in range(n_timesteps_post):
        x, y = data_post[i,:]
        mx = (x-xmin)//dx
        my = (y-ymin)//dy
        bin_numbers_post[i] = int(nx*my + mx)


    # Split data into training and testing sets:
    if test_on == 'pre':
        bin_numbers_use = bin_numbers_pre
    elif test_on == 'post':
        bin_numbers_use = bin_numbers_post
    else:
        print('Invalid test_on value. Must be "pre" or "post".')
        return None, None, None

    err_kfold = np.zeros(kCV)
    predicted_position = np.zeros((len(bin_numbers_use), 2))

    # 10-fold cross-validation
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_fold = {
            executor.submit(process_fold_floorflip, kk, data_pre, data_post, data_spks_pre, data_spks_post, bin_numbers_pre, bin_numbers_post, n_squares, boundaries, [], kCV, train_on, test_on, rotate): kk
            for kk in range(kCV)
        }

        for future in concurrent.futures.as_completed(future_to_fold):
            kk = future_to_fold[future]
            fold_error, fold_predicted_position = future.result()
            err_kfold[kk] = fold_error
            predicted_position += fold_predicted_position


    print('Spatial decoding error: ', 
        str(np.mean(err_kfold)) + ' +/- ' + str(np.std(err_kfold)))
    
    return err_kfold, None, predicted_position



#___________Decoding without floor flip______________________

def process_fold(kk, data_pos, data_spks, bin_numbers, n_squares, boundaries, bins_nodata, kCV):
    """
    Function to process each cross-validation fold independently.
    """

    batch_size = 1000  # Adjust this value based on available memory


    xmin, xmax, ymin, ymax = boundaries
    nx, ny = n_squares
    n_locations = nx * ny
    n_timesteps = len(data_pos)
    
    # Split the data into train and test for this fold
    bin_numbers_train, bin_numbers_test = cv_split(bin_numbers, kk, kCV)
    data_spks_train, data_spks_test = cv_split(data_spks, kk, kCV)
    data_pos_train, data_pos_test = cv_split(data_pos, kk, kCV)
    _, indices_test = cv_split(np.arange(n_timesteps), kk, kCV)

    # Train the classifiers for this fold
    clf_dict = {}
    bins_nodata = []
    
    def train_pairwise_classifier(i, j):
        bins_i_train = bin_numbers_train[bin_numbers_train == i]
        spks_i_train = data_spks_train[bin_numbers_train == i, :]
        bins_j_train = bin_numbers_train[bin_numbers_train == j]
        spks_j_train = data_spks_train[bin_numbers_train == j, :]
        
        if len(bins_i_train) == 0:
            return (i, None)
        elif len(bins_j_train) == 0:
            return (j, None)
        else:
            X_train = np.vstack((spks_i_train, spks_j_train))
            y_train = np.concatenate((np.ones_like(bins_i_train), np.zeros_like(bins_j_train)))
            clf = svm.SVC(kernel='linear', class_weight='balanced', C=0.1)
            clf.fit(X_train, y_train)
            return ((i, j), clf)

    # Train classifiers in a single thread
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            result = train_pairwise_classifier(i, j)
            if result[1] is None:
                bins_nodata.append(result[0])
            else:
                clf_dict[result[0]] = result[1]

    # Decode the positions for the test data
    def decode_batch(spks_batch, n_locations, clf_dict, bins_nodata):
        """
        Decode positions for a batch of spike count vectors.
        spks_batch: A batch of spike count vectors (2D array)
        n_locations: The number of spatial bins (locations)
        clf_dict: The dictionary containing trained classifiers
        bins_nodata: List of spatial bins without data
        """
        batch_size = len(spks_batch)  # Number of spike count vectors in the batch
        batch_votes = np.zeros((batch_size, n_locations))  # Initialize voting array

        # Loop over all pairs of locations (i, j) and make predictions
        for i in range(n_locations):
            for j in range(i + 1, n_locations):
                if (i not in bins_nodata) and (j not in bins_nodata):
                    clf = clf_dict.get((i, j), None)
                    if clf:
                        preds = clf.predict(spks_batch)  # Batch prediction
                        batch_votes[:, i] += preds
                        batch_votes[:, j] += 1 - preds
        return batch_votes


    # Decode all positions in the test set
    votes = np.zeros((len(data_spks_test), n_locations))

    for batch_start in range(0, len(data_spks_test), batch_size):
        spks_batch = data_spks_test[batch_start:batch_start+batch_size]
        batch_votes = decode_batch(spks_batch, n_locations, clf_dict, bins_nodata)
        votes[batch_start:batch_start+batch_size, :] = batch_votes
    
   
    # Predicted bins and positions
    bin_pred = np.argmax(votes, axis=1)
    pos_pred = np.zeros((len(bin_pred), 2))
    err = np.zeros(len(bin_pred))

    for i in range(len(bin_pred)):
        pos_pred[i, 0] = xmin + (0.5 + bin_pred[i] % nx) * (xmax - xmin) / nx
        pos_pred[i, 1] = ymin + (0.5 + bin_pred[i] // nx) * (ymax - ymin) / ny
        err[i] = np.linalg.norm(data_pos_test[i, :] - pos_pred[i, :])

    fold_error = np.median(err)
    predicted_positions = np.zeros((n_timesteps, 2))
    predicted_positions[indices_test, :] = pos_pred

    return fold_error, predicted_positions


def decoding_err_multithreadCV(data_pos, data_spks, n_squares, boundaries, kCV=10, shuffle=False, floor_flip=0):
    """
    Main function for decoding spatial position from spiking activity.
    """
    n_timesteps, n_neurons = np.shape(data_spks)
    print('n_neurons: ', n_neurons)

    nx, ny = n_squares
    xmin, xmax, ymin, ymax = boundaries

    # Shuffle the data if required
    if shuffle:
        roll_int = np.random.randint(n_timesteps // 10, n_timesteps - n_timesteps // 10)
        data_pos = np.roll(data_pos[::-1], roll_int)

    # Assign bins to positions
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    bin_numbers = np.zeros(n_timesteps)
    for i in range(n_timesteps):
        x, y = data_pos[i, :]
        mx = (x - xmin) // dx
        my = (y - ymin) // dy
        bin_numbers[i] = int(nx * my + mx)

    # Entropy calculation (as before)
    p_bin = np.array([np.sum(bin_numbers == i) / n_timesteps for i in range(int(nx * ny))])
    entropy = -np.sum(p_bin * np.log(1e-9 + p_bin))

    # Parallelize the k-fold cross-validation using ThreadPoolExecutor
    err_kfold = np.zeros(kCV)
    predicted_position = np.zeros((len(bin_numbers), 2))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_fold = {
            executor.submit(process_fold, kk, data_pos, data_spks, bin_numbers, n_squares, boundaries, [], kCV): kk
            for kk in range(kCV)
        }

        for future in concurrent.futures.as_completed(future_to_fold):
            kk = future_to_fold[future]
            fold_error, fold_predicted_position = future.result()
            err_kfold[kk] = fold_error
            predicted_position += fold_predicted_position  # This will accumulate over the folds

    print('Spatial decoding error: ', 
        str(np.mean(err_kfold)) + ' +/- ' + str(np.std(err_kfold)))

    return err_kfold, entropy, predicted_position


