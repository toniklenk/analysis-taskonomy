clear
clc
close all

RESULTS_PATH = './results taskonomy';
SAVE_PATH = './plots';


MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};


for model = 1:length(MODEL_NAMES)
    load(fullfile(RESULTS_PATH, ['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")
    %how many studies
    nst=4;
    
    
    % figure
    close all
    hf1=figure('position',[1,1,1500,1500], 'unit','pixel');
    set(gcf,'PaperUnits','centimeters','PaperSize',[20,20],'PaperPosition',[0,0,20,20])
    set(gcf,'color','w');
    set(0,'DefaultAxesFontName', 'Arial')
    set(0,'DefaultTextFontname', 'Arial')
    set(0,'DefaultAxesFontSize',12)
    set(0,'DefaultTextFontSize',12)
    sgtitle(MODEL_NAMES{model})
    
    %colormap
    cm=[1,0.15,1;0.4,0.4,1;...
        0.75*[1,0.15,1;0.4,0.4,1]];
    
    % plot lines
    
    %choose dnn stage to plot
    stages=1:5;
    
    %loop across integration and similarity measures
    dtx=0;
    for dt=1:3
        dti=dt;
        
        %loop across studies
        for st=1:nst
            dtx=dtx+1;
            
            for stage=stages
           
                %pick subplot
                subplot(3,nst,st+(dt-1)*nst);
                hold on
    
                %y axis
                y=dat.c{st}{dti}{stage}';
    
                %x axis
                x=1:length(y);
    
                %plot zero line
                h=line([x(1)-0.5,x(end)+0.5],[0,0]);
                set(h,'color','k');
                set(h,'linewidth',1.5);
                set(h,'linestyle','-');
    
                %fill area
                h=fill([x,x(end:-1:1)],[y,zeros(size(y))],(1-0.16*(6-stage))*cm(dt,:));
                set(h,'FaceAlpha',0.3);
                set(h,'linestyle','none');
    
                %plot line
                h=plot(x,y);
                set(h,'color',(1-0.16*(6-stage))*cm(dt,:));
                set(h,'linewidth',1.5);
    
                %plot properties
                ylim([-0.6,0.9]);
                xlim([x(1)-0.5,x(end)+0.5]);
                set(gca,'linewidth',1.5);
                %set(gca,'xtick',[]);
                %set(gca,'ytick',[]);
                set(gca,'xtick',[1,10,20,30, 40, 49]);
                
                xlab=xlabel('dnn layer');
    %             if dt==1
    %                 set(xlab,'visible','off');
    %             end
                if dt ==1
                    ylab=ylabel('integration-beauty');
                elseif dt == 2
                    ylab=ylabel('selfsimilarity-beauty');
                elseif dt == 3
                    ylab=ylabel('integration-beauty\newlinepartialed out selfsimilarity');
                end

                
                if st>1
                    set(ylab,'visible','off');
                end

                

    
                if st==1
                    at=title('Study 1\newlineShort Presentation');
                elseif st==2
                    at=title('Study 2\newlineLong Presentation');
                elseif st==3
                    at=title('Study 3\newlineRate_ComplexityOrder');
                elseif st==4
                    at=title('Study 4\newlineOasis Beauty Ratings');
                end
                set(at,'fontweight','normal');
                set(at,'verticalalignment','bottom');
                
                %signficance
                sig=dat.p{st}{dti}{stage}';
                
                for xi=1:length(x)
                    if sig(xi)<0.05/length(x)
                        yl=get(gca,'ylim');
                        h=plot(x(xi),yl(2)-0.05-0.025*(stage-1));
                        set(h,'marker','o')
                        set(h,'markersize',4)
                        set(h,'markeredgecolor',(1-0.16*(6-stage))*cm(dt,:));
                        set(h,'markerfacecolor',(1-0.16*(6-stage))*cm(dt,:));
                    end
                end
            
            end
        end
    end
    
    print(gcf,'-dtiff','-r200','fig_dnn_stages')

    export_fig(fullfile(SAVE_PATH, [MODEL_NAMES{model} '.png']));
    clf;
end