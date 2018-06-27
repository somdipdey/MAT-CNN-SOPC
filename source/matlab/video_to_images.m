videosDir = '/Users/somdipdey/Documents/MATLAB/Add-Ons/Collections/Deep Learning Tutorial Series/code/AHS/curated_video'
currDir = pwd
dinfo = dir(videosDir);
dinfo(ismember( {dinfo.name}, {'.', '..'})) = [];  %remove . and ..
%dirFlags = [dinfo.isdir] & ~strcmp({dinfo.name},'.') & ~strcmp({dinfo.name},'..');


for k= 1:length(dinfo)
    fprintf('Loop started.\n');
    if ~dinfo(k).isdir
        continue
    end
    path = strcat(videosDir,'/',dinfo(k).name)

    currDir
    dinfo(k)
    videoPath=strcat(path,'/medium_2.mp4')
  %  shuttleVideo = audiovideo.mmreader(videoPath);
    shuttleVideo = VideoReader(videoPath);
    
    frVidInfo=get(shuttleVideo);
    numberOfFrames = shuttleVideo.NumberOfFrames
    vidHeight = shuttleVideo.Height;
    vidWidth = shuttleVideo.Width;
    framesToRead = 1:12:numberOfFrames
 % initialize a matrix for all frames to be read
   % allFrames = zeros(vidHeight, vidWidth, 3, length(framesToRead));

    img_dir = strcat('img/',dinfo(k).name)
    path = fullfile(currDir,img_dir)
    if exist(path, 'dir')

         if length(dir([path '/*.jpg']))>1
             continue
%         else
%            delete(path,'\*.fig')
         end
    else
        mkdir(path)
    end
 % read in the frames
    for k=1:length(framesToRead)

       frameIdx = framesToRead(k);

       currentFrame   = read(shuttleVideo,frameIdx);
       filename = sprintf('%d.jpg',k-1);
      % filename = [sprintf('%03d',ii) '.jpg'];
       fullname = fullfile(currDir,img_dir,filename);
       imwrite(currentFrame,fullname);

%        if k==1
%            % cast the all frames matrix appropriately
%            allFrames = cast(allFrames, class(currentFrame));
%        end
% 
%        allFrames(:,:,:,k) = currentFrame;
    end
    
end
 

% Code will end adnormally beacuse it will search video in images folder as
% well !!!
